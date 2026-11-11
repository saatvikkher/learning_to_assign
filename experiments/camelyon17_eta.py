"""
Experiment: Camelyon17 — Eta Sensitivity for Tree Greedy

Sweeps eta across a range of values for the Tree Greedy policy only,
using the auto-selected best hospital pair with sim_samples=2000.

Usage:
    python -m experiments.camelyon17_eta [options]

    --etas          Comma-separated eta values  (default: 0.0,0.1,0.5,1.0,2.0,5.0)
    --n-runs        Runs per (eta, capacity)    (default: 100)
    --capacities    Weaker-agent fractions      (default: 0.2,0.4,0.5,0.6,0.8)
    --sim-samples   Experiment pool size        (default: 2000)
    --hospital-a    Center index for agent A    (default: -1 = auto)
    --hospital-b    Center index for agent B    (default: -1 = auto)
    --ood-center    OOD center index            (default: -1 = auto)
    --plot          Generate plot from most recent log and exit
    --log-file      Specific log file (used with --plot)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import run_online
from experiments.camelyon17 import (
    load_or_build_feature_cache,
    train_hospital_models,
    analyze_pairs,
    prepare_experiment_data,
    METADATA_PATH,
)


def make_eta_plot(log_path: Path, out_path: Path | None = None) -> Path:
    boundaries: dict[float, float] = {}
    runs: dict[float, dict[float, list[float]]] = {}  # eta -> cap -> [error_rates]

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["run"] == -1:
                boundaries[rec["capacity"]] = rec["error_rate"]
            else:
                eta = float(rec["eta"])
                cap = float(rec["capacity"])
                runs.setdefault(eta, {}).setdefault(cap, []).append(rec["error_rate"])

    if not runs:
        raise ValueError(f"No run data in {log_path}")

    if out_path is None:
        out_path = log_path.parent.parent / "plots" / (log_path.stem + ".pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eta_values = sorted(runs.keys())
    colors = cm.plasma(np.linspace(0.1, 0.82, len(eta_values)))

    fig, ax = plt.subplots(figsize=(14, 6))

    for color, eta in zip(colors, eta_values):
        caps = sorted(runs[eta].keys())
        xs = [1.0 - c for c in caps]
        ys = [float(np.mean(runs[eta][c])) for c in caps]
        if 0.0 in boundaries:
            xs = [1.0] + xs
            ys = [boundaries[0.0]] + ys
        if 1.0 in boundaries:
            xs = xs + [0.0]
            ys = ys + [boundaries[1.0]]
        order = np.argsort(xs)[::-1]
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        ax.plot(xs, ys, color=color, marker="o", markersize=6, linewidth=2,
                label=f"$\\eta$ = {eta}")

    # Random: straight line between marginals (never empirical)
    if 0.0 in boundaries and 1.0 in boundaries:
        ax.plot([1.0, 0.0], [boundaries[0.0], boundaries[1.0]],
                color="black", linestyle="--", linewidth=2, label="Non-contextual")

    ax.set_xlabel("Capacity (Agent 1)", fontsize=13)
    ax.set_ylabel("Error Rate", fontsize=13)
    # place legend to the right so it doesn't overlap the plot
    leg = ax.legend(fontsize=11, frameon=True, loc="center left",
                    bbox_to_anchor=(1.02, 0.5))
    leg.get_frame().set_alpha(1.0)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_facecolor("white")
    ax.grid(alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # leave room on the right for the legend
    fig.tight_layout(rect=(0, 0, 0.85, 1))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_experiment(args: argparse.Namespace) -> Path:
    df = pd.read_csv(METADATA_PATH, index_col=0)
    print(f"Metadata: {len(df)} patches, {df['center'].nunique()} centers")

    print("Loading/building feature cache...")
    all_features = load_or_build_feature_cache(df, n_workers=args.n_workers)

    print("Training per-hospital models (split=0 data)...")
    models = train_hospital_models(df, all_features, seed=args.seed)

    print("Analyzing hospital pairs...")
    pair_results = analyze_pairs(df, all_features, models, seed=args.seed)
    best = pair_results[0]
    center_a = args.hospital_a if args.hospital_a >= 0 else best["ca"]
    center_b = args.hospital_b if args.hospital_b >= 0 else best["cb"]

    all_centers = sorted(df["center"].unique())
    ood_candidates = [c for c in all_centers if c not in (center_a, center_b)]
    if args.ood_center >= 0:
        ood_center = args.ood_center
    elif ood_candidates:
        sizes = {c: ((df["center"] == c) & (df["split"] == 1)).sum() for c in ood_candidates}
        ood_center = min(sizes, key=sizes.get)
    else:
        ood_center = None

    print(f"\nExperiment setup: center_a={center_a}  center_b={center_b}  ood={ood_center}")

    print("Preparing experiment pool (split=1 data)...")
    X, rewards_matrix, acc0, acc1, name0, name1 = prepare_experiment_data(
        df, all_features, center_a, center_b, models,
        ood_center=ood_center, sim_samples=args.sim_samples, seed=args.seed,
    )
    n = len(X)
    print(f"  Pool size: {n}")
    print(f"  Agent 0 (weaker):   {name0}  acc={acc0:.4f}  err={1-acc0:.4f}")
    print(f"  Agent 1 (stronger): {name1}  acc={acc1:.4f}  err={1-acc1:.4f}")

    eta_values = [float(e) for e in args.etas.split(",")]
    capacities = [float(c) for c in args.capacities.split(",")]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"eta_sensitivity_camelyon17_{timestamp}.jsonl"
    log_path = log_dir / log_name
    print(f"Logging to {log_path}")

    total = len(eta_values) * len(capacities) * args.n_runs
    print(f"  {len(eta_values)} eta values × {len(capacities)} capacities × {args.n_runs} runs = {total} simulations")

    meta = {
        "experiment": "eta_sensitivity_camelyon17",
        "center_a": int(center_a),
        "center_b": int(center_b),
        "ood_center": int(ood_center) if ood_center is not None else None,
        "agent0": name0,
        "agent1": name1,
        "seed": args.seed,
        "sim_samples": args.sim_samples,
    }

    done = 0
    t0 = time.time()

    with open(log_path, "w") as f:
        # Boundary conditions: marginal error rates, independent of eta
        f.write(json.dumps({"run": -1, "capacity": 0.0, "error_rate": 1.0 - acc1, **meta}) + "\n")
        f.write(json.dumps({"run": -1, "capacity": 1.0, "error_rate": 1.0 - acc0, **meta}) + "\n")
        f.flush()

        # Outer loop: run_idx — full family of curves plottable after any completed iteration
        for run_idx in range(args.n_runs):
            # Single permutation shared across all eta values in this run
            perm_rng = np.random.default_rng([args.seed, run_idx, 0])
            perm = perm_rng.permutation(n)

            for eta in eta_values:
                for cap in capacities:
                    p_targets = np.array([cap, 1.0 - cap])
                    # algo_idx=3 (tree_greedy position in main experiment) → +1 = 4
                    algo_rng = np.random.default_rng([args.seed, run_idx, 4])
                    acc = run_online(X, rewards_matrix, p_targets, perm, "tree_greedy", eta, rng=algo_rng)

                    rec = {
                        "run": run_idx,
                        "eta": eta,
                        "capacity": cap,
                        "error_rate": 1.0 - acc,
                        "ts": datetime.utcnow().isoformat(),
                        **meta,
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()

                    done += 1
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (total - done) / rate if rate > 0 else 0
                    print(
                        f"  run {run_idx+1}/{args.n_runs}  eta={eta}  cap={cap}"
                        f"  [{done}/{total} | {remaining/60:.1f} min left]",
                        end="\r",
                    )
            print()

    print(f"\nDone. Log: {log_path}")
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Eta sensitivity experiment (Tree Greedy, Camelyon17)")
    parser.add_argument("--hospital-a", type=int, default=-1, help="Center index for agent A (-1 = auto)")
    parser.add_argument("--hospital-b", type=int, default=-1, help="Center index for agent B (-1 = auto)")
    parser.add_argument("--ood-center", type=int, default=-1, help="OOD center index (-1 = auto)")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--etas", default="0.0,0.1,0.5,1.0,2.0,5.0")
    parser.add_argument("--capacities", default="0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--sim-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel workers for feature extraction")
    parser.add_argument("--plot", action="store_true", help="Plot from most recent log and exit")
    parser.add_argument("--log-file", default=None, help="Specific log file to plot")
    args = parser.parse_args()

    if args.plot:
        if args.log_file is None:
            logs = sorted(Path(args.log_dir).glob("eta_sensitivity_camelyon17_*.jsonl"))
            if not logs:
                print("No eta_sensitivity_camelyon17 log files found.")
                sys.exit(1)
            log_file = logs[-1]
        else:
            log_file = Path(args.log_file)
        out = make_eta_plot(log_file)
        print(f"Plot saved to {out}")
        return

    log_path = run_experiment(args)
    print("Generating plot...")
    out = make_eta_plot(log_path)
    print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
