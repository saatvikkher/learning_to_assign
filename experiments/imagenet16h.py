"""
Experiment: ImageNet-16h — Human vs Machine Classifier

Usage:
    python -m experiments.imagenet16h [options]

    --epoch       Model epoch to use: baseline, 00, 01, 10  (default: 10)
    --model       Model name in dataset (default: densenet161)
                  Options: alexnet, densenet161, googlenet, resnet152, vgg19
    --participant Participant ID                              (default: 34)
    --n-runs      Number of random-seed runs per capacity    (default: 100)
    --capacities  Comma-separated weaker-agent capacity fractions (default: 0.2,0.4,0.5,0.6,0.8)
    --dup         Data duplication factor                    (default: 10)
    --eta         Lagrangian step size                       (default: 0.5)
    --seed        Base random seed                           (default: 42)
    --log-dir     Directory for JSONL logs                   (default: logs/)
    --plot        Generate plot from completed log and exit
    --log-file    Read a specific log file (used with --plot)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import run_online, run_offline, train_offline_models
from src.plotting import make_plot

DATA_DIR = ROOT / "data" / "imagenet16h"
HUMAN_PATH = DATA_DIR / "Behavioral Data" / "Preprocessed" / "human_only_classification_6per_img_preprocessed.csv"
MODEL_PATH_TEMPLATE = DATA_DIR / "Machine Classifier Predictions" / "Preprocessed" / "hai_epoch{epoch}_model_preds_max_unnormalized_preprocessed.csv"
BASELINE_PATH = DATA_DIR / "Machine Classifier Predictions" / "Preprocessed" / "hai_baseline_model_preds_max_unnormalized_preprocessed.csv"


def load_data(epoch: str, participant_id: int, dup_factor: int, model_name: str = "densenet161") -> tuple:
    """
    Returns (X, rewards_matrix, acc_agent0, acc_agent1, agent0_name, agent1_name)
    where Agent 0 is always the weaker agent (higher error rate).
    """
    if epoch == "baseline":
        model_path = BASELINE_PATH
    else:
        model_path = Path(str(MODEL_PATH_TEMPLATE).format(epoch=epoch))

    df_human = pd.read_csv(HUMAN_PATH)
    df_model = pd.read_csv(model_path)

    df_human = df_human[df_human["participant_id"] == participant_id]
    df_model = df_model[df_model["model_name"] == model_name]

    df = pd.merge(
        df_human[["image_name", "noise_level", "correct"]],
        df_model[["image_name", "noise_level", "correct"]],
        on=["image_name", "noise_level"],
        suffixes=("_human", "_model"),
    )

    if dup_factor > 1:
        df = pd.concat([df] * dup_factor, ignore_index=True)

    noise = df["noise_level"].values.reshape(-1, 1)
    X = np.column_stack([noise, np.ones(len(noise))])

    r_human = df["correct_human"].values.astype(float)
    r_model = df["correct_model"].values.astype(float)

    acc_human = r_human.mean()
    acc_model = r_model.mean()

    # Agent 0 = weaker (lower accuracy), Agent 1 = stronger
    model_label = model_name
    if acc_human <= acc_model:
        rewards_matrix = np.column_stack([r_human, r_model])
        return X, rewards_matrix, acc_human, acc_model, f"Human-{participant_id}", model_label
    else:
        rewards_matrix = np.column_stack([r_model, r_human])
        return X, rewards_matrix, acc_model, acc_human, model_label, f"Human-{participant_id}"


def log_boundary(f, capacity: float, error_rate: float, meta: dict) -> None:
    rec = {"capacity": capacity, "run": -1, "error_rate": error_rate, **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


def log_run(f, capacity: float, run: int, policies: dict[str, float], meta: dict) -> None:
    rec = {"capacity": capacity, "run": run, "policies": policies, "ts": datetime.utcnow().isoformat(), **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


def run_experiment(args: argparse.Namespace) -> Path:
    print(f"Loading data  epoch={args.epoch}  model={args.model}  participant={args.participant}  dup={args.dup}")
    X, rewards_matrix, acc0, acc1, name0, name1 = load_data(args.epoch, args.participant, args.dup, args.model)
    print(f"  Agent 0 (weaker):  {name0}  acc={acc0:.4f}")
    print(f"  Agent 1 (stronger): {name1}  acc={acc1:.4f}")
    print(f"  Samples: {len(X)}")

    capacities = [float(c) for c in args.capacities.split(",")]

    # Build log path
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"imagenet16h_ep{args.epoch}_{args.model}_p{args.participant}_{timestamp}.jsonl"
    log_path = log_dir / log_name
    print(f"Logging to {log_path}")

    meta = {
        "experiment": "imagenet16h",
        "epoch": args.epoch,
        "model": args.model,
        "participant": args.participant,
        "agent0": name0,
        "agent1": name1,
        "eta": args.eta,
        "dup": args.dup,
        "seed": args.seed,
    }

    # Train offline models on full data (no permutation)
    print("Training offline baselines...")
    offline_models = train_offline_models(X, rewards_matrix[:, 0], rewards_matrix[:, 1], seed=args.seed)

    # Unconstrained offline accuracy (no capacity penalty)
    base_perm = np.arange(len(X))
    unc_log_acc = run_offline(X, rewards_matrix, np.zeros(2), base_perm, "offline_logistic_unconstrained", offline_models, args.eta)
    unc_tree_acc = run_offline(X, rewards_matrix, np.zeros(2), base_perm, "offline_tree_unconstrained", offline_models, args.eta)
    print(f"  Unconstrained Logistic acc={unc_log_acc:.4f}  Tree acc={unc_tree_acc:.4f}")

    with open(log_path, "w") as f:
        # Write boundary conditions (capacity of weaker agent = 0 or 1)
        # cap_weaker=0 → stronger gets everything → error = 1 - acc1
        log_boundary(f, 0.0, 1.0 - acc1, meta)
        # cap_weaker=1 → weaker gets everything → error = 1 - acc0
        log_boundary(f, 1.0, 1.0 - acc0, meta)

        # Unconstrained offline as special "capacity=-1" entries (used by plotting as axhlines)
        log_boundary(f, -1.0, 1.0 - unc_log_acc, {**meta, "policy": "Unconstrained Logistic"})
        log_boundary(f, -1.0, 1.0 - unc_tree_acc, {**meta, "policy": "Unconstrained Tree"})

        # Main loop — outer over run_idx so each pass fills one point per capacity,
        # allowing the full curve to be plotted after any completed iteration.
        total = len(capacities) * args.n_runs
        done = 0
        t0 = time.time()

        for run_idx in range(args.n_runs):
            # Dedicated permutation seed — completely separate from algorithm seeds.
            # All algorithms receive the same perm so they see identical point orderings.
            perm_rng = np.random.default_rng([args.seed, run_idx, 0])
            perm = perm_rng.permutation(len(X))

            for cap in capacities:
                p_targets = np.array([cap, 1.0 - cap])

                policies: dict[str, float] = {}

                # Offline constrained (deterministic given perm — no RNG needed)
                policies["Offline Logistic"] = run_offline(X, rewards_matrix, p_targets, perm, "offline_logistic", offline_models, args.eta)
                policies["Offline Tree"] = run_offline(X, rewards_matrix, p_targets, perm, "offline_tree", offline_models, args.eta)

                # Online — each policy gets a fresh RNG seeded independently of perm.
                # Using [seed, run_idx, algo_idx] ensures policies don't share state.
                for algo_idx, (mode, label) in enumerate([
                    ("log_ts",      "Logistic TS"),
                    ("log_greedy",  "Logistic Greedy"),
                    ("tree_ts",     "Tree TS"),
                    ("tree_greedy", "Tree Greedy"),
                    ("random",      "Random"),
                ]):
                    algo_rng = np.random.default_rng([args.seed, run_idx, algo_idx + 1])
                    policies[label] = run_online(X, rewards_matrix, p_targets, perm, mode, args.eta, rng=algo_rng)

                log_run(f, cap, run_idx, policies, meta)

                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate if rate > 0 else 0
                print(f"  run {run_idx+1}/{args.n_runs}  cap={cap}  [{done}/{total} total | {remaining/60:.1f} min left]", end="\r")

            print()

    print(f"\nDone. Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet-16h routing experiment")
    parser.add_argument("--epoch", default="10", choices=["baseline", "00", "01", "10"])
    parser.add_argument("--model", default="densenet161",
                        choices=["alexnet", "densenet161", "googlenet", "resnet152", "vgg19"])
    parser.add_argument("--participant", type=int, default=34)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--capacities", default="0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--dup", type=int, default=10)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--plot", action="store_true", help="Plot from existing log and exit")
    parser.add_argument("--log-file", default=None, help="Log file to plot (used with --plot)")
    args = parser.parse_args()

    if args.plot:
        if args.log_file is None:
            # Find most recent log
            log_dir = Path(args.log_dir)
            logs = sorted(log_dir.glob("imagenet16h_*.jsonl"))
            if not logs:
                print("No log files found.")
                sys.exit(1)
            log_file = logs[-1]
        else:
            log_file = Path(args.log_file)
        out = make_plot(log_file)
        print(f"Plot saved to {out}")
        return

    log_path = run_experiment(args)
    print("Generating plot...")
    out = make_plot(log_path)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
