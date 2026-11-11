"""
Experiment: MMLU — Cohere Small vs LLaMA-13B

Two LLMs with complementary expertise across college_chemistry and us_foreign_policy.
Context is a 1D PCA of sentence embeddings (all-MiniLM-L6-v2) plus a bias term.

Usage:
    python -m experiments.mmlu [options]

    --model-a     First model (default: cohere/medium-20221108)
    --model-b     Second model (default: meta/llama-2-70b)
    --cat1        First MMLU category (default: college_chemistry)
    --cat2        Second MMLU category (default: us_foreign_policy)
    --n-runs      Number of random-seed runs per capacity    (default: 100)
    --capacities  Comma-separated weaker-agent capacity fractions (default: 0.2,0.4,0.5,0.6,0.8)
    --eta         Lagrangian step size                       (default: 0.5)
    --seed        Base random seed                           (default: 42)
    --log-dir     Directory for JSONL logs                   (default: logs/)
    --plot        Generate plot from most recent log and exit
    --log-file    Specific log file to plot (used with --plot)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import run_online, run_offline, train_offline_models
from src.plotting import make_plot

DATA_PATH = ROOT / "data" / "mmlu_scored_progress.csv"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_data(model_a: str, model_b: str, cat1: str, cat2: str) -> tuple:
    """
    Returns (X, rewards_matrix, acc0, acc1, name0, name1)
    where Agent 0 is always the weaker agent (lower accuracy).
    Context X is [1D-PCA-of-embedding, 1.0], shape (n, 2).
    """
    from sentence_transformers import SentenceTransformer
    import torch

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["prompt", "category", "model"])

    exp_df = df[df["category"].isin([cat1, cat2]) & df["model"].isin([model_a, model_b])]
    pivot = exp_df.pivot(index="prompt", columns="model", values="is_correct").dropna()

    prompts = pivot.index.tolist()
    rewards_a = pivot[model_a].values.astype(float)
    rewards_b = pivot[model_b].values.astype(float)

    # Sentence embeddings → 1D PCA + bias
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Encoding {len(prompts)} prompts on {device}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL, device=device)
    embeddings = embedder.encode(prompts, batch_size=64, show_progress_bar=False)
    embedder.cpu()
    del embedder
    if device == "cuda":
        torch.cuda.empty_cache()

    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(embeddings)
    X = np.column_stack([X_pca, np.ones(len(X_pca))])

    acc_a = rewards_a.mean()
    acc_b = rewards_b.mean()

    # Agent 0 = weaker (lower accuracy)
    if acc_a <= acc_b:
        rewards_matrix = np.column_stack([rewards_a, rewards_b])
        return X, rewards_matrix, acc_a, acc_b, model_a, model_b
    else:
        rewards_matrix = np.column_stack([rewards_b, rewards_a])
        return X, rewards_matrix, acc_b, acc_a, model_b, model_a


def log_boundary(f, capacity: float, error_rate: float, meta: dict) -> None:
    rec = {"capacity": capacity, "run": -1, "error_rate": error_rate, **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


def log_run(f, capacity: float, run: int, policies: dict[str, float], meta: dict) -> None:
    rec = {"capacity": capacity, "run": run, "policies": policies, "ts": datetime.utcnow().isoformat(), **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


def run_experiment(args: argparse.Namespace) -> Path:
    print(f"Loading data  model_a={args.model_a}  model_b={args.model_b}")
    print(f"  Categories: {args.cat1}  {args.cat2}")
    X, rewards_matrix, acc0, acc1, name0, name1 = load_data(
        args.model_a, args.model_b, args.cat1, args.cat2
    )
    print(f"  Agent 0 (weaker):   {name0}  acc={acc0:.4f}  err={1-acc0:.4f}")
    print(f"  Agent 1 (stronger): {name1}  acc={acc1:.4f}  err={1-acc1:.4f}")
    print(f"  Samples: {len(X)}")

    capacities = [float(c) for c in args.capacities.split(",")]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"mmlu_{timestamp}.jsonl"
    log_path = log_dir / log_name
    print(f"Logging to {log_path}")

    meta = {
        "experiment": "mmlu",
        "model_a": args.model_a,
        "model_b": args.model_b,
        "cat1": args.cat1,
        "cat2": args.cat2,
        "agent0": name0,
        "agent1": name1,
        "eta": args.eta,
        "seed": args.seed,
    }

    print("Training offline baselines...")
    offline_models = train_offline_models(X, rewards_matrix[:, 0], rewards_matrix[:, 1], seed=args.seed)

    base_perm = np.arange(len(X))
    unc_log_acc = run_offline(X, rewards_matrix, np.zeros(2), base_perm, "offline_logistic_unconstrained", offline_models, args.eta)
    unc_tree_acc = run_offline(X, rewards_matrix, np.zeros(2), base_perm, "offline_tree_unconstrained", offline_models, args.eta)
    print(f"  Unconstrained Logistic acc={unc_log_acc:.4f}  Tree acc={unc_tree_acc:.4f}")

    with open(log_path, "w") as f:
        log_boundary(f, 0.0, 1.0 - acc1, meta)
        log_boundary(f, 1.0, 1.0 - acc0, meta)
        log_boundary(f, -1.0, 1.0 - unc_log_acc, {**meta, "policy": "Unconstrained Logistic"})
        log_boundary(f, -1.0, 1.0 - unc_tree_acc, {**meta, "policy": "Unconstrained Tree"})

        # Outer loop over run_idx so each pass fills one point per capacity —
        # the full curve is plottable after any completed iteration.
        total = len(capacities) * args.n_runs
        done = 0
        t0 = time.time()

        for run_idx in range(args.n_runs):
            # Dedicated permutation seed, independent of algorithm seeds
            perm_rng = np.random.default_rng([args.seed, run_idx, 0])
            perm = perm_rng.permutation(len(X))

            for cap in capacities:
                p_targets = np.array([cap, 1.0 - cap])

                policies: dict[str, float] = {}

                policies["Offline Logistic"] = run_offline(X, rewards_matrix, p_targets, perm, "offline_logistic", offline_models, args.eta)
                policies["Offline Tree"] = run_offline(X, rewards_matrix, p_targets, perm, "offline_tree", offline_models, args.eta)

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
    parser = argparse.ArgumentParser(description="MMLU LLM routing experiment")
    parser.add_argument("--model-a", default="cohere/small-20220720")
    parser.add_argument("--model-b", default="meta/llama-13b")
    parser.add_argument("--cat1", default="college_chemistry")
    parser.add_argument("--cat2", default="us_foreign_policy")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--capacities", default="0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--plot", action="store_true", help="Plot from most recent log and exit")
    parser.add_argument("--log-file", default=None, help="Specific log file to plot")
    args = parser.parse_args()

    if args.plot:
        if args.log_file is None:
            log_dir = Path(args.log_dir)
            logs = sorted(log_dir.glob("mmlu_*.jsonl"))
            if not logs:
                print("No mmlu log files found.")
                sys.exit(1)
            log_file = logs[-1]
        else:
            log_file = Path(args.log_file)
        out = make_plot(log_file, xlabel="Capacity (Agent 1 — stronger)")
        print(f"Plot saved to {out}")
        return

    log_path = run_experiment(args)
    print("Generating plot...")
    out = make_plot(log_path, xlabel="Capacity (Agent 1 — stronger)")
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
