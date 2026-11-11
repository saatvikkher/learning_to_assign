"""
Experiment: per-step assignment/reward trajectories at a single capacity split.

Runs the usual policy set on camelyon17, mmlu, imagenet16h or civilcomments
at one capacity combination — by default (0.2, 0.8), i.e. Agent 0 (weaker) gets 0.2 and
Agent 1 (stronger) gets 0.8 — and records, for every step of every run, which
agent the task was assigned to and the reward that assignment realized.

Nothing here replaces the main runners: data loading, the Agent-0-is-weaker
convention, the RNG discipline and the JSONL log format are all reused from
them, so the average rewards logged here match what the corresponding main
experiment logs at the same capacity, seed and run index.

Usage:
    python -m experiments.step_trajectories --dataset imagenet16h [options]

    --dataset         camelyon17 | mmlu | imagenet16h | civilcomments  (required)
    --n-runs          Number of runs per capacity            (default: 100)
    --capacities      Comma-separated weaker-agent capacities (default: 0.2)
    --eta             Lagrangian step size                   (default: 0.5)
    --seed            Base random seed                       (default: 42)
    --log-dir         Directory for JSONL logs               (default: logs/)
    --traj-dir        Directory for trajectory arrays        (default: trajectories/)
    --snapshot-every  Snapshot cadence, in runs              (default: 5)
    --snapshot-start  First run index to snapshot at         (default: 5)
    --workers         Parallel processes over runs            (default: 1)

    ImageNet-16h:  --epoch --model --participant --dup
    MMLU:          --model-a --model-b --cat1 --cat2
    Camelyon17:    --hospital-a --hospital-b --ood-center --sim-samples --n-workers
    CivilComments: --cc-split --cc-model-a --cc-model-b --pooled --pca-dim --balance-seed

Outputs (stem = <dataset>_<timestamp>):
    logs/steptraj_<stem>.jsonl              per-run average rewards (plottable)
    trajectories/<stem>_step_actions.npy    uint8 (n_caps, n_policies, n_runs, n_steps)
    trajectories/<stem>_step_rewards.npy    uint8, same shape
    trajectories/<stem>_perms.npy           int32 (n_runs, n_steps) arrival order
    trajectories/<stem>_meta.json           axis labels + experiment metadata
    trajectories/<stem>_demographic.npy      per-row subpopulation (civilcomments only)

The three arrays are rewritten (atomically) every --snapshot-every runs, each
snapshot truncated to the runs completed so far, so an interrupted job still
leaves a usable, self-describing result.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import train_offline_models
from src.simulation_traj import run_online_traj, run_offline_traj

# Offline policies are deterministic given perm; the online ones are enumerated
# separately so algo_idx (and therefore the RNG seed) matches the main runners.
OFFLINE_POLICIES = [
    ("offline_logistic", "Offline Logistic"),
    ("offline_tree", "Offline Tree"),
]
ONLINE_POLICIES = [
    ("log_ts", "Logistic TS"),
    ("log_greedy", "Logistic Greedy"),
    ("tree_ts", "Tree TS"),
    ("tree_greedy", "Tree Greedy"),
    ("random", "Random"),
]
POLICY_LABELS = [label for _, label in OFFLINE_POLICIES + ONLINE_POLICIES]


# ---------------------------------------------------------------------------
# Dataset loading — delegates to the existing experiment modules
# ---------------------------------------------------------------------------

def load_dataset(args: argparse.Namespace) -> tuple:
    """
    Returns (X, rewards_matrix, acc0, acc1, name0, name1, meta, extras).

    ``extras`` holds any per-row side arrays worth saving alongside the
    trajectories (e.g. the civilcomments subpopulation of each item); each is
    written to ``<stem>_<key>.npy`` and is indexed like the rows of X, so
    ``perms[run]`` maps a step onto it too.
    """
    if args.dataset == "imagenet16h":
        from experiments.imagenet16h import load_data

        print(f"Loading imagenet16h  epoch={args.epoch}  model={args.model}  "
              f"participant={args.participant}  dup={args.dup}")
        X, rewards, acc0, acc1, name0, name1 = load_data(
            args.epoch, args.participant, args.dup, args.model
        )
        meta = {
            "experiment": "imagenet16h",
            "epoch": args.epoch,
            "model": args.model,
            "participant": args.participant,
            "dup": args.dup,
        }
        return X, rewards, acc0, acc1, name0, name1, meta, {}

    if args.dataset == "mmlu":
        from experiments.mmlu import load_data

        print(f"Loading mmlu  model_a={args.model_a}  model_b={args.model_b}")
        print(f"  Categories: {args.cat1}  {args.cat2}")
        X, rewards, acc0, acc1, name0, name1 = load_data(
            args.model_a, args.model_b, args.cat1, args.cat2
        )
        meta = {
            "experiment": "mmlu",
            "model_a": args.model_a,
            "model_b": args.model_b,
            "cat1": args.cat1,
            "cat2": args.cat2,
        }
        return X, rewards, acc0, acc1, name0, name1, meta, {}

    if args.dataset == "camelyon17":
        import pandas as pd
        from experiments import camelyon17 as cam

        df = pd.read_csv(cam.METADATA_PATH, index_col=0)
        print(f"Loading camelyon17  metadata: {len(df)} patches, "
              f"{df['center'].nunique()} centers")
        all_features = cam.load_or_build_feature_cache(df, n_workers=args.n_workers)

        print("Training per-hospital models (split=0 data)...")
        models = cam.train_hospital_models(df, all_features, seed=args.seed)

        center_a, center_b = args.hospital_a, args.hospital_b
        if center_a < 0 or center_b < 0:
            print("Analyzing hospital pairs...")
            pair_results = cam.analyze_pairs(df, all_features, models, seed=args.seed)
            cam.print_analysis(pair_results)
            best = pair_results[0]
            center_a = args.hospital_a if args.hospital_a >= 0 else best["ca"]
            center_b = args.hospital_b if args.hospital_b >= 0 else best["cb"]

        # Auto-select OOD center exactly as the main runner does: the center
        # outside the pair with the fewest test samples.
        all_centers = sorted(df["center"].unique())
        ood_candidates = [c for c in all_centers if c not in (center_a, center_b)]
        if args.ood_center >= 0:
            ood_center = args.ood_center
        elif ood_candidates:
            sizes = {c: ((df["center"] == c) & (df["split"] == 1)).sum() for c in ood_candidates}
            ood_center = min(sizes, key=sizes.get)
        else:
            ood_center = None

        print(f"  Center A: {center_a}  |  Center B: {center_b}  |  OOD: {ood_center}")
        X, rewards, acc0, acc1, name0, name1 = cam.prepare_experiment_data(
            df, all_features, center_a, center_b, models,
            ood_center=ood_center, sim_samples=args.sim_samples, seed=args.seed,
        )
        meta = {
            "experiment": "camelyon17",
            "center_a": int(center_a),
            "center_b": int(center_b),
            "ood_center": int(ood_center) if ood_center is not None else None,
        }
        return X, rewards, acc0, acc1, name0, name1, meta, {}

    if args.dataset == "civilcomments":
        from experiments.civilcomments import load_data

        print(f"Loading civilcomments  split={args.cc_split}  "
              f"model_a={args.cc_model_a}  model_b={args.cc_model_b}  pooled={args.pooled}")
        X, rewards, acc0, acc1, name0, name1, demo = load_data(
            args.cc_split, args.cc_model_a, args.cc_model_b, args.pooled,
            args.pca_dim, args.balance_seed, args.seed
        )
        for d in sorted(set(demo)):
            m = demo == d
            print(f"    subpop {d:8s} n={m.sum():5d}  "
                  f"agent0 acc={rewards[m, 0].mean():.4f}  agent1 acc={rewards[m, 1].mean():.4f}")
        # "split" rather than "dataset" so the key does not collide with the
        # runner's own --dataset in the manifest.
        meta = {
            "experiment": "civilcomments",
            "split": args.cc_split,
            "pooled": args.pooled,
            "pca_dim": args.pca_dim,
            "balance_seed": args.balance_seed,
            "model_a": args.cc_model_a,
            "model_b": args.cc_model_b,
        }
        return X, rewards, acc0, acc1, name0, name1, meta, {"demographic": np.asarray(demo, dtype=str)}

    raise ValueError(f"Unknown dataset: {args.dataset}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_boundary(f, capacity: float, error_rate: float, meta: dict) -> None:
    rec = {"capacity": capacity, "run": -1, "error_rate": error_rate, **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


def log_run(f, capacity: float, run: int, policies: dict[str, float], meta: dict) -> None:
    rec = {
        "capacity": capacity,
        "run": run,
        "policies": policies,
        "ts": datetime.utcnow().isoformat(),
        **meta,
    }
    f.write(json.dumps(rec) + "\n")
    f.flush()


def save_array(path: Path, arr: np.ndarray) -> None:
    """Write via a temporary file so an interrupted snapshot never truncates one."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)


def save_snapshot(
    traj_dir: Path,
    stem: str,
    actions: np.ndarray,
    rewards: np.ndarray,
    perms: np.ndarray,
    runs_done: int,
    manifest: dict,
) -> None:
    """Write arrays truncated to the completed runs, plus the manifest."""
    save_array(traj_dir / f"{stem}_step_actions.npy", actions[:, :, :runs_done, :])
    save_array(traj_dir / f"{stem}_step_rewards.npy", rewards[:, :, :runs_done, :])
    save_array(traj_dir / f"{stem}_perms.npy", perms[:runs_done])

    manifest = {**manifest, "n_runs_completed": runs_done,
                "updated": datetime.now().isoformat(timespec="seconds")}
    tmp = traj_dir / f"{stem}_meta.json.tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, traj_dir / f"{stem}_meta.json")


# ---------------------------------------------------------------------------
# One run = one task
#
# A task is a whole run (all capacities, all policies), so results still come
# back run-by-run and every snapshot covers a contiguous prefix of runs. Since
# each generator is seeded from (seed, run_idx, algo_idx) alone, tasks are
# independent and the output is identical to the serial loop whatever
# --workers is set to.
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(X, rewards_matrix, offline_models, capacities, eta, seed) -> None:
    _W.update(X=X, rewards_matrix=rewards_matrix, offline_models=offline_models,
              capacities=capacities, eta=eta, seed=seed)


def _run_task(run_idx: int) -> tuple:
    """Returns (run_idx, perm, policies_per_cap, actions, rewards)."""
    X, rewards_matrix = _W["X"], _W["rewards_matrix"]
    capacities, eta, seed = _W["capacities"], _W["eta"], _W["seed"]
    n = len(X)

    # Dedicated permutation seed, shared by every policy in this run.
    perm = np.random.default_rng([seed, run_idx, 0]).permutation(n)

    actions = np.zeros((len(capacities), len(POLICY_LABELS), n), dtype=np.uint8)
    rewards = np.zeros((len(capacities), len(POLICY_LABELS), n), dtype=np.uint8)
    policies_per_cap = []

    for cap_idx, cap in enumerate(capacities):
        p_targets = np.array([cap, 1.0 - cap])
        policies: dict[str, float] = {}
        pol_idx = 0

        for mode, label in OFFLINE_POLICIES:
            avg, acts, rews = run_offline_traj(
                X, rewards_matrix, p_targets, perm, mode, _W["offline_models"], eta
            )
            policies[label] = avg
            actions[cap_idx, pol_idx] = acts
            rewards[cap_idx, pol_idx] = rews
            pol_idx += 1

        # Fresh RNG per policy, seeded as in the main runners so the averages
        # reproduce those runs exactly.
        for algo_idx, (mode, label) in enumerate(ONLINE_POLICIES):
            algo_rng = np.random.default_rng([seed, run_idx, algo_idx + 1])
            avg, acts, rews = run_online_traj(
                X, rewards_matrix, p_targets, perm, mode, eta, rng=algo_rng
            )
            policies[label] = avg
            actions[cap_idx, pol_idx] = acts
            rewards[cap_idx, pol_idx] = rews
            pol_idx += 1

        policies_per_cap.append(policies)

    return run_idx, perm, policies_per_cap, actions, rewards


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> Path:
    X, rewards_matrix, acc0, acc1, name0, name1, data_meta, extras = load_dataset(args)
    n = len(X)
    print(f"  Agent 0 (weaker):   {name0}  acc={acc0:.4f}  err={1-acc0:.4f}")
    print(f"  Agent 1 (stronger): {name1}  acc={acc1:.4f}  err={1-acc1:.4f}")
    print(f"  Samples: {n}")

    capacities = [float(c) for c in args.capacities.split(",")]
    print("  Capacity splits (Agent 0, Agent 1): "
          + ", ".join(f"({c:.2f}, {1-c:.2f})" for c in capacities))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.dataset}_{timestamp}"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"steptraj_{stem}.jsonl"

    traj_dir = Path(args.traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logging to {log_path}")
    print(f"Trajectories to {traj_dir}/{stem}_step_{{actions,rewards}}.npy")

    meta = {
        **data_meta,
        "agent0": name0,
        "agent1": name1,
        "eta": args.eta,
        "seed": args.seed,
    }

    print("Training offline baselines...")
    offline_models = train_offline_models(
        X, rewards_matrix[:, 0], rewards_matrix[:, 1], seed=args.seed
    )

    n_caps = len(capacities)
    n_pol = len(POLICY_LABELS)
    # (capacity, policy, run, step) — uint8 is enough for 2 agents / 0-1 rewards
    step_actions = np.zeros((n_caps, n_pol, args.n_runs, n), dtype=np.uint8)
    step_rewards = np.zeros((n_caps, n_pol, args.n_runs, n), dtype=np.uint8)
    perms = np.zeros((args.n_runs, n), dtype=np.int32)

    manifest = {
        **meta,
        "dataset": args.dataset,
        "acc_agent0": float(acc0),
        "acc_agent1": float(acc1),
        "n_steps": n,
        "n_runs_requested": args.n_runs,
        "capacities": capacities,
        "capacity_note": "capacity[i] is Agent 0's (weaker) target share; "
                         "Agent 1 (stronger) gets 1 - capacity[i]",
        "policies": POLICY_LABELS,
        "axes": ["capacity", "policy", "run", "step"],
        "step_order": "arrival order; perms[run] maps step -> row of X",
        "log_file": str(log_path),
        "extra_arrays": sorted(extras),
    }

    for key, arr in extras.items():
        save_array(traj_dir / f"{stem}_{key}.npy", np.asarray(arr))

    with open(log_path, "w") as f:
        log_boundary(f, 0.0, 1.0 - acc1, meta)
        log_boundary(f, 1.0, 1.0 - acc0, meta)

        total = args.n_runs
        done = 0
        t0 = time.time()

        init_args = (X, rewards_matrix, offline_models, capacities, args.eta, args.seed)
        if args.workers > 1:
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=init_args)
        else:
            _init_worker(*init_args)
            pool = None

        try:
            # Runs are dispatched in blocks of --workers and each block is
            # awaited in full, so the completed runs are always a prefix and a
            # snapshot can simply truncate.
            block = max(args.workers, 1)
            last_snapshot = 0
            for start in range(0, args.n_runs, block):
                run_ids = list(range(start, min(start + block, args.n_runs)))
                results = pool.map(_run_task, run_ids) if pool else [_run_task(i) for i in run_ids]

                for run_idx, perm, policies_per_cap, acts, rews in results:
                    perms[run_idx] = perm
                    step_actions[:, :, run_idx, :] = acts
                    step_rewards[:, :, run_idx, :] = rews
                    for cap_idx, cap in enumerate(capacities):
                        log_run(f, cap, run_idx, policies_per_cap[cap_idx], meta)

                    done += 1
                    elapsed = time.time() - t0
                    remaining = (total - done) * elapsed / done
                    print(f"  run {run_idx+1}/{args.n_runs}  "
                          f"[{done}/{total} | {remaining/60:.1f} min left]", flush=True)

                # Comparing against the last snapshot (rather than a modulo)
                # keeps the cadence right when a block spans several intervals.
                runs_done = run_ids[-1] + 1
                if (runs_done >= args.snapshot_start
                        and runs_done - last_snapshot >= args.snapshot_every):
                    save_snapshot(traj_dir, stem, step_actions, step_rewards,
                                  perms, runs_done, manifest)
                    last_snapshot = runs_done
                    print(f"  snapshot saved at run {runs_done}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    save_snapshot(traj_dir, stem, step_actions, step_rewards, perms, args.n_runs, manifest)
    print(f"\nDone. Log: {log_path}")
    print(f"Trajectories: {traj_dir}/{stem}_step_actions.npy "
          f"(shape {(n_caps, n_pol, args.n_runs, n)})")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-step assignment/reward trajectories at a single capacity split"
    )
    parser.add_argument("--dataset", required=True,
                        choices=["camelyon17", "mmlu", "imagenet16h", "civilcomments"])
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--capacities", default="0.2",
                        help="Weaker-agent (Agent 0) capacity shares; 0.2 means the (0.2, 0.8) split")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--traj-dir", default="trajectories")
    parser.add_argument("--snapshot-every", type=int, default=5)
    parser.add_argument("--snapshot-start", type=int, default=5)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes over runs; results are seed-identical")

    # ImageNet-16h (defaults = the canonical Participant 48 vs VGG19 pair)
    parser.add_argument("--epoch", default="10", choices=["baseline", "00", "01", "10"])
    parser.add_argument("--model", default="vgg19",
                        choices=["alexnet", "densenet161", "googlenet", "resnet152", "vgg19"])
    parser.add_argument("--participant", type=int, default=48)
    parser.add_argument("--dup", type=int, default=10)

    # MMLU
    parser.add_argument("--model-a", default="cohere/small-20220720")
    parser.add_argument("--model-b", default="meta/llama-13b")
    parser.add_argument("--cat1", default="college_chemistry")
    parser.add_argument("--cat2", default="us_foreign_policy")

    # Camelyon17
    parser.add_argument("--hospital-a", type=int, default=-1, help="-1 = auto-select")
    parser.add_argument("--hospital-b", type=int, default=-1, help="-1 = auto-select")
    parser.add_argument("--ood-center", type=int, default=-1, help="-1 = auto-select")
    parser.add_argument("--sim-samples", type=int, default=2000)
    parser.add_argument("--n-workers", type=int, default=8)

    # CivilComments (defaults = the canonical pooled race split)
    parser.add_argument("--cc-split", default="race", choices=["race", "gender"])
    parser.add_argument("--cc-model-a", default="meta_llama-7b")
    parser.add_argument("--cc-model-b", default="together_redpajama-incite-instruct-3b-v1")
    parser.add_argument("--pooled", action="store_true",
                        help="Pool all 5 text perturbations (the canonical setting)")
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--balance-seed", type=int, default=0)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
