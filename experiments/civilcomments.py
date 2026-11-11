"""
Experiment: CivilComments — routing toxicity classification across demographic
subpopulations (subpopulation shift, WILDS framing).

Two LLMs with complementary expertise across the two demographic identities
mentioned in the comment (black/white for the race split, female/male for the
gender split). Context is a 10D PCA of sentence embeddings (all-MiniLM-L6-v2)
plus a bias term — 1D PCA recovers the subpopulation at only AUC~0.58 here,
vs ~0.83 at 10D, so the 1D context used for MMLU is too weak.

Labels are balanced within each subpopulation: the raw toxic rate is 29% (race)
and 14% (gender), which makes near-constant "not toxic" predictors look like the
strongest models and hides all genuine expertise differences.

Usage:
    python -m experiments.civilcomments [options]

    --dataset     race | gender                              (default: race)
    --model-a     First model  (default: meta_llama-7b)
    --model-b     Second model (default: together_redpajama-incite-instruct-3b-v1)
    --pooled      Pool all 5 text perturbations (5x more items)
    --pca-dim     PCA components in the context              (default: 10)
    --balance-seed  Seed for the label-balancing subsample    (default: 0)
    --n-runs      Number of random-seed runs per capacity     (default: 100)
    --capacities  Comma-separated capacity fractions          (default: 0.2,0.4,0.5,0.6,0.8)
    --policies    Comma-separated subset of policy labels to run (default: all)
    --eta         Lagrangian step size                        (default: 0.5)
    --seed        Base random seed                            (default: 42)
    --workers     Parallel processes over (run, capacity) tasks (default: 1)
    --log-dir     Directory for JSONL logs                    (default: logs/)
    --plot        Generate plot from most recent log and exit
    --log-file    Specific log file to plot (used with --plot)
"""

import argparse
import json
import multiprocessing
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

DATA_DIR = ROOT / "data" / "CivilComments"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ONLINE_POLICIES = [
    ("log_ts", "Logistic TS"),
    ("log_greedy", "Logistic Greedy"),
    ("tree_ts", "Tree TS"),
    ("tree_greedy", "Tree Greedy"),
    ("random", "Random"),
]


def _embeddings(dataset: str, records: list) -> dict:
    """Embed every comment once per dataset and cache to disk."""
    cache = DATA_DIR / f"emb_{dataset}_minilm.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(zip(z["ids"].tolist(), z["emb"]))

    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Encoding {len(records)} comments on {device}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL, device=device)
    emb = embedder.encode([r["input_text"] for r in records], batch_size=128,
                          show_progress_bar=False)
    embedder.cpu()
    del embedder
    if device == "cuda":
        torch.cuda.empty_cache()

    ids = np.array([r["id"] for r in records])
    np.savez_compressed(cache, ids=ids, emb=np.asarray(emb))
    return dict(zip(ids.tolist(), np.asarray(emb)))


def balance_labels(items: pd.DataFrame, seed: int) -> np.ndarray:
    """Item ids with the majority label downsampled within each subpopulation."""
    rng = np.random.default_rng(seed)
    keep = []
    for _, cell in items.groupby("demographic"):
        n = cell["label"].value_counts().min()
        for _, grp in cell.groupby("label"):
            keep.append(rng.permutation(grp["id"].values)[:n])
    return np.concatenate(keep)


def load_data(dataset: str, model_a: str, model_b: str, pooled: bool,
              pca_dim: int, balance_seed: int, seed: int = 42) -> tuple:
    """
    Returns (X, rewards_matrix, acc0, acc1, name0, name1, demographic)
    where Agent 0 is always the weaker agent (lower balanced accuracy).
    Context X is [pca_dim-D PCA of embedding, 1.0], shape (n, pca_dim + 1).
    """
    records = json.load(open(DATA_DIR / f"{dataset}_combined.json"))
    emb = _embeddings(dataset, records)

    rows = []
    for r in records:
        if not pooled and r["perturbation"] is not None:
            continue
        ra, rb = r["models"].get(model_a), r["models"].get(model_b)
        if not ra or not rb or ra.get("prediction") is None or rb.get("prediction") is None:
            continue
        rows.append({
            "id": r["id"], "demographic": r["demographic"], "label": bool(r["reference"]),
            "r_a": float(bool(ra["prediction"]) == bool(r["reference"])),
            "r_b": float(bool(rb["prediction"]) == bool(r["reference"])),
        })
    df = pd.DataFrame(rows)
    df = df[df["id"].isin(balance_labels(df, balance_seed))].reset_index(drop=True)

    E = np.stack([emb[i] for i in df["id"]])
    X_pca = PCA(n_components=pca_dim, random_state=seed).fit_transform(E)
    X = np.column_stack([X_pca, np.ones(len(X_pca))])

    acc_a, acc_b = df["r_a"].mean(), df["r_b"].mean()
    if acc_a <= acc_b:
        rewards = df[["r_a", "r_b"]].values
        return X, rewards, acc_a, acc_b, model_a, model_b, df["demographic"].values
    rewards = df[["r_b", "r_a"]].values
    return X, rewards, acc_b, acc_a, model_b, model_a, df["demographic"].values


def log_boundary(f, capacity: float, error_rate: float, meta: dict) -> None:
    rec = {"capacity": capacity, "run": -1, "error_rate": error_rate, **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


def log_run(f, capacity: float, run: int, policies: dict[str, float], meta: dict) -> None:
    rec = {"capacity": capacity, "run": run, "policies": policies,
           "ts": datetime.utcnow().isoformat(), **meta}
    f.write(json.dumps(rec) + "\n")
    f.flush()


# --- Parallel execution -------------------------------------------------------
# One task = one (run_idx, capacity) pair. Because every generator is seeded
# from (seed, run_idx, algo_idx) alone, tasks are independent and the results
# are identical to the serial loop regardless of worker count or ordering.

_W: dict = {}


def _init_worker(X, rewards_matrix, online, offline_models, run_offline_constrained,
                 eta, seed) -> None:
    _W.update(X=X, rewards_matrix=rewards_matrix, online=online,
              offline_models=offline_models,
              run_offline_constrained=run_offline_constrained, eta=eta, seed=seed)


def _run_task(task: tuple[int, float]) -> tuple[int, float, dict[str, float]]:
    """Run every selected policy for one (run_idx, capacity) pair."""
    run_idx, cap = task
    X, rewards_matrix = _W["X"], _W["rewards_matrix"]
    eta, seed = _W["eta"], _W["seed"]

    perm = np.random.default_rng([seed, run_idx, 0]).permutation(len(X))
    p_targets = np.array([cap, 1.0 - cap])
    policies: dict[str, float] = {}

    if _W["run_offline_constrained"]:
        policies["Offline Logistic"] = run_offline(
            X, rewards_matrix, p_targets, perm, "offline_logistic",
            _W["offline_models"], eta)
        policies["Offline Tree"] = run_offline(
            X, rewards_matrix, p_targets, perm, "offline_tree",
            _W["offline_models"], eta)

    for algo_idx, (mode, label) in enumerate(_W["online"]):
        algo_rng = np.random.default_rng([seed, run_idx, algo_idx + 1])
        policies[label] = run_online(X, rewards_matrix, p_targets, perm,
                                     mode, eta, rng=algo_rng)

    return run_idx, cap, policies


def run_experiment(args: argparse.Namespace) -> Path:
    print(f"Loading data  dataset={args.dataset}  model_a={args.model_a}  model_b={args.model_b}")
    X, rewards_matrix, acc0, acc1, name0, name1, demo = load_data(
        args.dataset, args.model_a, args.model_b, args.pooled, args.pca_dim,
        args.balance_seed, args.seed
    )
    print(f"  Agent 0 (weaker):   {name0}  acc={acc0:.4f}  err={1-acc0:.4f}")
    print(f"  Agent 1 (stronger): {name1}  acc={acc1:.4f}  err={1-acc1:.4f}")
    for d in sorted(set(demo)):
        m = demo == d
        print(f"    subpop {d:8s} n={m.sum():5d}  agent0 acc={rewards_matrix[m, 0].mean():.4f}  "
              f"agent1 acc={rewards_matrix[m, 1].mean():.4f}")
    print(f"  Samples: {len(X)}   context dim: {X.shape[1]}")

    capacities = [float(c) for c in args.capacities.split(",")]
    selected = None if args.policies is None else set(args.policies.split(","))
    online = [(m, l) for m, l in ONLINE_POLICIES if selected is None or l in selected]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"civilcomments_{args.dataset}_{timestamp}.jsonl"
    print(f"Logging to {log_path}")

    meta = {
        "experiment": "civilcomments",
        "dataset": args.dataset,
        "pooled": args.pooled,
        "pca_dim": args.pca_dim,
        "balance_seed": args.balance_seed,
        "model_a": args.model_a, "model_b": args.model_b,
        "agent0": name0, "agent1": name1,
        "eta": args.eta, "seed": args.seed,
    }

    offline_models = None
    if selected is None or {"Offline Logistic", "Offline Tree",
                            "Unconstrained Logistic", "Unconstrained Tree"} & selected:
        print("Training offline baselines...")
        offline_models = train_offline_models(X, rewards_matrix[:, 0], rewards_matrix[:, 1],
                                              seed=args.seed)

    with open(log_path, "w") as f:
        log_boundary(f, 0.0, 1.0 - acc1, meta)
        log_boundary(f, 1.0, 1.0 - acc0, meta)
        if offline_models is not None:
            base_perm = np.arange(len(X))
            for mode, label in [("offline_logistic_unconstrained", "Unconstrained Logistic"),
                                ("offline_tree_unconstrained", "Unconstrained Tree")]:
                a = run_offline(X, rewards_matrix, np.zeros(2), base_perm, mode,
                                offline_models, args.eta)
                print(f"  {label} acc={a:.4f}")
                log_boundary(f, -1.0, 1.0 - a, {**meta, "policy": label})

        tasks = [(run_idx, cap) for run_idx in range(args.n_runs) for cap in capacities]
        total = len(tasks)
        done, t0 = 0, time.time()

        run_offline_constrained = offline_models is not None and (
            selected is None or {"Offline Logistic", "Offline Tree"} & selected)
        init_args = (X, rewards_matrix, online, offline_models,
                     run_offline_constrained, args.eta, args.seed)

        if args.workers > 1:
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=init_args)
            results = pool.imap_unordered(_run_task, tasks, chunksize=1)
        else:
            _init_worker(*init_args)
            pool = None
            results = map(_run_task, tasks)

        try:
            for run_idx, cap, policies in results:
                log_run(f, cap, run_idx, policies, meta)
                done += 1
                elapsed = time.time() - t0
                remaining = (total - done) * elapsed / done
                print(f"  run {run_idx+1}/{args.n_runs}  cap={cap}  "
                      f"[{done}/{total} | {remaining/60:.1f} min left]", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    print(f"\nDone. Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CivilComments LLM routing experiment")
    parser.add_argument("--dataset", default="race", choices=["race", "gender"])
    parser.add_argument("--model-a", default="meta_llama-7b")
    parser.add_argument("--model-b", default="together_redpajama-incite-instruct-3b-v1")
    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--balance-seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--capacities", default="0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--policies", default=None)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes over (run, capacity) tasks")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.plot:
        if args.log_file is None:
            logs = sorted(Path(args.log_dir).glob("civilcomments_*.jsonl"))
            if not logs:
                print("No civilcomments log files found.")
                sys.exit(1)
            log_file = logs[-1]
        else:
            log_file = Path(args.log_file)
        out = make_plot(log_file, xlabel="Capacity (Agent 1 — stronger)")
        print(f"Plot saved to {out}")
        return

    log_path = run_experiment(args)


if __name__ == "__main__":
    main()
