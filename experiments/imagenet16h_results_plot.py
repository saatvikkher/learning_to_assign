"""
Results plot for ImageNet-16h.
Computes offline unconstrained baselines (trained on the same test pool as the
online experiment) and renders the styled results plot.

Usage:
    python -m experiments.imagenet16h_results_plot [--log-file logs/imagenet16h_*.jsonl]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from experiments.imagenet16h import load_data
from src.plotting import load_log, make_styled_results_plot


def compute_offline_unconstrained(log_path: Path) -> tuple[dict, list]:
    """
    Reconstruct the test pool from the log metadata, train one logistic and
    one RF per agent's reward (same hyperparams as online), route to argmax,
    and return (error_rates_dict, assignment_ratios).
    """
    import json
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("run", -1) >= 0:
                seed     = rec["seed"]
                epoch    = rec["epoch"]
                model    = rec["model"]
                part     = rec["participant"]
                dup      = rec["dup"]
                break

    X, rewards, _, _, _, _ = load_data(epoch, part, dup, model)
    n = len(X)

    log1 = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, rewards[:, 0])
    log2 = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, rewards[:, 1])
    s0_lr = log1.predict_proba(X)[:, 1]
    s1_lr = log2.predict_proba(X)[:, 1]
    routing_lr = (s1_lr >= s0_lr).astype(int)
    err_lr  = 1 - rewards[np.arange(n), routing_lr].mean()
    ratio_lr = routing_lr.mean()

    rf1 = RandomForestRegressor(n_estimators=20, max_depth=3, min_samples_leaf=10,
                                random_state=seed).fit(X, rewards[:, 0])
    rf2 = RandomForestRegressor(n_estimators=20, max_depth=3, min_samples_leaf=10,
                                random_state=seed).fit(X, rewards[:, 1])
    routing_rf = (rf2.predict(X) >= rf1.predict(X)).astype(int)
    err_rf  = 1 - rewards[np.arange(n), routing_rf].mean()
    ratio_rf = routing_rf.mean()

    print(f"  Offline Unc. Logistic error={err_lr:.4f}  ratio_agent1={ratio_lr:.4f}")
    print(f"  Offline Unc. Tree     error={err_rf:.4f}  ratio_agent1={ratio_rf:.4f}")
    return {"Logistic": err_lr, "Tree": err_rf}, [ratio_lr, ratio_rf]


def make_imagenet_plot(log_path: Path) -> Path:
    print("Loading log...")
    log_data = load_log(log_path)

    print("Computing offline unconstrained baselines...")
    offline_unc, ratios = compute_offline_unconstrained(log_path)

    out_path = log_path.parent.parent / "plots" / (log_path.stem + ".pdf")
    out = make_styled_results_plot(log_data, offline_unc, ratios, out_path)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        logs = sorted((ROOT / "logs").glob("imagenet16h_*.jsonl"))
        if not logs:
            print("No imagenet16h log files found.")
            sys.exit(1)
        log_path = logs[-1]

    out = make_imagenet_plot(log_path)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
