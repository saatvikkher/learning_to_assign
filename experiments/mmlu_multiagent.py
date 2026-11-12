"""
Experiment: multi-agent LLM routing on MMLU (A = 3 or A = 5 agents).

The capacity of one focus agent (LLaMa-2-70B) is varied; the remaining capacity
is split equally across the other agents. Context is the first principal
component of all-MiniLM-L6-v2 question embeddings plus a bias term, as in the
two-agent MMLU experiment.

Usage:
    python -m experiments.mmlu_multiagent --setting 3agent --workers 25
    python -m experiments.mmlu_multiagent --setting 5agent --workers 25
    python -m experiments.mmlu_multiagent --plot --log-file logs/mmlu_3agent_<timestamp>.jsonl

Options:
    --setting     3agent | 5agent (agents and categories of the paper)  (default: 3agent)
    --n-runs      Task streams per capacity                             (default: 25)
    --capacities  Capacities of the focus agent to simulate             (default: 0.0,0.2,0.4,0.5,0.6,0.8)
    --eta         Queue penalty                                         (default: 0.5)
    --seed        Base random seed                                      (default: 42)
    --workers     Parallel processes over (run, capacity) pairs         (default: 1)
    --log-dir     Directory for JSONL logs                              (default: logs/)

Capacity 1.0 (all tasks to the focus agent) is the focus agent's marginal error
and is logged as a boundary, not simulated.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import run_online

DATA_PATH = ROOT / "data" / "mmlu_scored_progress.csv"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

FOCUS = "meta/llama-2-70b"
SETTINGS = {
    "3agent": {
        "agents": ["cohere/small-20220720", "meta/llama-13b", "meta/llama-2-70b"],
        "cats": ["college_chemistry", "computer_security", "us_foreign_policy"],
    },
    "5agent": {
        "agents": ["cohere/small-20220720", "meta/llama-13b", "meta/llama-2-70b",
                   "cohere/command-medium-beta", "together/t0pp"],
        "cats": ["abstract_algebra", "college_chemistry", "computer_security",
                 "econometrics", "us_foreign_policy"],
    },
}
LABELS = {
    "cohere/small-20220720": "Cohere Small",
    "meta/llama-13b": "LLaMa-13B",
    "meta/llama-2-70b": "LLaMa-2-70B",
    "cohere/command-medium-beta": "command-medium",
    "together/t0pp": "t0pp",
}

# Same order (and therefore the same algo_rng seeds) as experiments/mmlu.py
ONLINE_POLICIES = [
    ("log_ts", "Logistic TS"),
    ("log_greedy", "Logistic Greedy"),
    ("tree_ts", "Tree TS"),
    ("tree_greedy", "Tree Greedy"),
    ("random", "Random"),
]


def load_data(agents: list[str], cats: list[str], seed: int) -> tuple:
    """Returns (X, rewards_matrix) with columns in the order of `agents`.
    Context X is [1D-PCA-of-embedding, 1.0], shape (n, 2)."""
    from sentence_transformers import SentenceTransformer

    df = pd.read_csv(DATA_PATH).drop_duplicates(subset=["prompt", "category", "model"])
    df = df[df["category"].isin(cats) & df["model"].isin(agents)]
    pivot = df.pivot(index="prompt", columns="model", values="is_correct").dropna()

    embedder = SentenceTransformer(EMBEDDER_MODEL, device="cpu")
    emb = embedder.encode(pivot.index.tolist(), batch_size=64, show_progress_bar=False)
    X_pca = PCA(n_components=1, random_state=seed).fit_transform(emb)
    X = np.column_stack([X_pca, np.ones(len(X_pca))])
    return X, pivot[agents].values.astype(float)


def targets_for(cap: float, n_agents: int, focus_idx: int) -> np.ndarray:
    p = np.full(n_agents, (1.0 - cap) / (n_agents - 1))
    p[focus_idx] = cap
    return p


def _run_task(X, R, focus_idx, cap, run_idx, eta, seed):
    perm = np.random.default_rng([seed, run_idx, 0]).permutation(len(X))
    p_targets = targets_for(cap, R.shape[1], focus_idx)
    out = {}
    for algo_idx, (mode, label) in enumerate(ONLINE_POLICIES):
        algo_rng = np.random.default_rng([seed, run_idx, algo_idx + 1])
        out[label] = run_online(X, R, p_targets, perm, mode, eta, rng=algo_rng)
    return cap, run_idx, out


def run_experiment(args: argparse.Namespace) -> Path:
    setting = SETTINGS[args.setting]
    agents, cats = setting["agents"], setting["cats"]
    focus_idx = agents.index(FOCUS)
    capacities = [float(c) for c in args.capacities.split(",")]

    print(f"Loading data  setting={args.setting}  agents={agents}")
    X, R = load_data(agents, cats, args.seed)
    acc = R.mean(axis=0)
    for a, m in zip(agents, acc):
        print(f"  {a:30s} acc={m:.4f}")
    print(f"  Tasks: {len(X)}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"mmlu_{args.setting}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    meta = {"experiment": f"mmlu_{args.setting}", "agents": agents, "cats": cats,
            "focus": FOCUS, "eta": args.eta, "seed": args.seed}

    tasks = [(cap, r) for r in range(args.n_runs) for cap in capacities]
    t0 = time.time()
    with open(log_path, "w") as f:
        f.write(json.dumps({"header": True, **meta, "n": len(X),
                            "accuracy": dict(zip(agents, acc.tolist()))}) + "\n")
        f.flush()

        def write(cap, run_idx, policies):
            f.write(json.dumps({"capacity": cap, "run": run_idx, "policies": policies,
                                "ts": datetime.utcnow().isoformat()}) + "\n")
            f.flush()

        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_run_task, X, R, focus_idx, c, r, args.eta, args.seed)
                        for c, r in tasks]
                for i, fu in enumerate(as_completed(futs), 1):
                    write(*fu.result())
                    print(f"  {i}/{len(tasks)}  [{(time.time() - t0) / 60:.1f} min]", end="\r")
        else:
            for i, (c, r) in enumerate(tasks, 1):
                write(*_run_task(X, R, focus_idx, c, r, args.eta, args.seed))
                print(f"  {i}/{len(tasks)}  [{(time.time() - t0) / 60:.1f} min]", end="\r")

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Log: {log_path}")
    return log_path


def make_plot(log_path: Path, legend: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import POLICY_COLOR

    runs: dict = {}
    for line in open(log_path):
        rec = json.loads(line)
        if rec.get("header"):
            header = rec
            continue
        for pol, r in rec["policies"].items():
            runs.setdefault(pol, {}).setdefault(rec["capacity"], []).append(1.0 - r)

    agents, focus = header["agents"], header["focus"]
    err = {a: 1.0 - header["accuracy"][a] for a in agents}
    err_focus = err[focus]
    err_others = np.mean([err[a] for a in agents if a != focus])

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # Non-contextual: assign to agent a with probability alpha_a, independent of x
    ax.plot([0.0, 1.0], [err_others, err_focus], color=POLICY_COLOR["Random"],
            linestyle="--", linewidth=2, label="Non-contextual")
    for pol in ["Logistic TS", "Logistic Greedy", "Tree TS", "Tree Greedy"]:
        caps = sorted(runs[pol])
        xs = caps + [1.0]
        ys = [np.mean(runs[pol][c]) for c in caps] + [err_focus]
        ax.plot(xs, ys, marker="o", linewidth=2, color=POLICY_COLOR[pol], label=pol)

    ax.set_xlim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel(f"Capacity of {LABELS.get(focus, focus)}", fontsize=18, fontweight="medium")
    ax.set_ylabel("Error Rate", fontsize=18, fontweight="medium")
    if legend:
        ax.legend(fontsize=11, frameon=False, ncol=2)
    fig.tight_layout()
    out = ROOT / "plots" / f"{Path(log_path).stem}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def summarize(log_path: Path) -> None:
    rows = []
    for line in open(log_path):
        rec = json.loads(line)
        if rec.get("header"):
            print(f"{rec['experiment']}: n={rec['n']}  accuracy=" +
                  ", ".join(f"{LABELS.get(a, a)} {v:.3f}" for a, v in rec["accuracy"].items()))
            continue
        for pol, r in rec["policies"].items():
            rows.append({"capacity": rec["capacity"], "policy": pol, "error": 1.0 - r})
    t = pd.DataFrame(rows).groupby(["policy", "capacity"])["error"].agg(["mean", "sem", "count"])
    print(t["mean"].unstack().round(4).to_string())
    print(f"max SE: {t['sem'].max():.4f}   runs per cell: {t['count'].min()}-{t['count'].max()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent MMLU routing experiment")
    parser.add_argument("--setting", default="3agent", choices=list(SETTINGS))
    parser.add_argument("--n-runs", type=int, default=25)
    parser.add_argument("--capacities", default="0.0,0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--plot", action="store_true", help="Plot an existing log and exit")
    parser.add_argument("--log-file", default=None, help="Log file to plot (with --plot)")
    parser.add_argument("--no-legend", action="store_true")
    args = parser.parse_args()

    if args.plot:
        log_file = Path(args.log_file) if args.log_file else \
            sorted(Path(args.log_dir).glob(f"mmlu_{args.setting}_*.jsonl"))[-1]
    else:
        log_file = run_experiment(args)
    summarize(log_file)
    print(f"Plot saved to {make_plot(log_file, legend=not args.no_legend)}")


if __name__ == "__main__":
    main()
