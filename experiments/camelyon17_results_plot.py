"""
Results plot for the Camelyon17 experiment.
Replaces the JSONL-cached unconstrained baselines with freshly computed
offline unconstrained policies (logistic + tree) trained on the same 2000-point
test pool as the online experiment, with the same hyperparameters.

Usage:
    python -m experiments.camelyon17_results_plot [--log-file logs/camelyon17_*.jsonl]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.plotting import load_log, make_styled_results_plot

DATA_ROOT  = ROOT / "data" / "camelyon17_v1.0"
META_PATH  = DATA_ROOT / "metadata.csv"
CACHE_PATH = DATA_ROOT / "features_cache.npz"


def compute_offline_unconstrained(log_path: Path) -> tuple[dict, list]:
    """
    Reconstruct the exact 2000-point test pool from the log metadata,
    train one logistic and one random forest (same hyperparams as the online
    agents) on the pool to predict each agent's reward, route to argmax with
    no capacity constraint, and return (error_rates_dict, assignment_ratios).
    """
    import json
    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("run", -1) >= 0:
                seed       = rec["seed"]
                center_a   = rec["center_a"]
                center_b   = rec["center_b"]
                ood_center = rec["ood_center"]
                break

    df       = pd.read_csv(META_PATH, index_col=0)
    cache    = np.load(CACHE_PATH)
    idx2pos  = {int(i): k for k, i in enumerate(cache["indices"])}
    feat_arr = cache["features"]

    mask      = df["center"].isin([center_a, center_b, ood_center]) & (df["split"] == 1)
    pool_df   = df[mask]
    pool_feats = feat_arr[[idx2pos[int(i)] for i in pool_df.index]]
    rng       = np.random.default_rng(seed)
    sel       = rng.choice(len(pool_df), 2000, replace=False)
    pool_feats = pool_feats[sel]
    labels    = pool_df["tumor"].values[sel]

    def train_hosp(center):
        rng2 = np.random.default_rng(seed)
        sub  = df[(df["center"] == center) & (df["split"] == 0)]
        X    = feat_arr[[idx2pos[int(i)] for i in sub.index]]
        y    = sub["tumor"].values
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        half = 7500
        s = np.concatenate([rng2.choice(pos, min(half, len(pos)), replace=False),
                            rng2.choice(neg, min(half, len(neg)), replace=False)])
        sc = StandardScaler()
        m  = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=seed)
        m.fit(sc.fit_transform(X[s]), y[s])
        return m, sc

    ma, sca = train_hosp(center_a)
    mb, scb = train_hosp(center_b)
    ca = (ma.predict(sca.transform(pool_feats)) == labels).astype(float)
    cb = (mb.predict(scb.transform(pool_feats)) == labels).astype(float)

    if ca.mean() <= cb.mean():
        rewards = np.column_stack([ca, cb])
    else:
        rewards = np.column_stack([cb, ca])

    X_ctx = np.column_stack([
        PCA(1, random_state=seed).fit_transform(
            StandardScaler().fit_transform(pool_feats)).ravel(),
        np.ones(2000),
    ])

    log1 = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X_ctx, rewards[:, 0])
    log2 = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X_ctx, rewards[:, 1])
    s0_lr = log1.predict_proba(X_ctx)[:, 1]
    s1_lr = log2.predict_proba(X_ctx)[:, 1]
    routing_lr = (s1_lr >= s0_lr).astype(int)
    err_lr  = 1 - rewards[np.arange(2000), routing_lr].mean()
    ratio_lr = routing_lr.mean()

    rf1 = RandomForestRegressor(n_estimators=20, max_depth=3, min_samples_leaf=10,
                                random_state=seed).fit(X_ctx, rewards[:, 0])
    rf2 = RandomForestRegressor(n_estimators=20, max_depth=3, min_samples_leaf=10,
                                random_state=seed).fit(X_ctx, rewards[:, 1])
    routing_rf = (rf2.predict(X_ctx) >= rf1.predict(X_ctx)).astype(int)
    err_rf  = 1 - rewards[np.arange(2000), routing_rf].mean()
    ratio_rf = routing_rf.mean()

    print(f"  Offline Unc. Logistic error={err_lr:.4f}  ratio_agent1={ratio_lr:.4f}")
    print(f"  Offline Unc. Tree     error={err_rf:.4f}  ratio_agent1={ratio_rf:.4f}")
    return {"Logistic": err_lr, "Tree": err_rf}, [ratio_lr, ratio_rf]


def make_camelyon17_plot(log_path: Path) -> Path:
    print("Loading log...")
    log_data = load_log(log_path)

    print("Computing offline unconstrained baselines...")
    offline_unc, ratios = compute_offline_unconstrained(log_path)

    out_path = log_path.parent.parent / "plots" / (log_path.stem + ".pdf")
    out = make_styled_results_plot(log_data, {"Logistic": offline_unc["Logistic"]}, ratios,
                                   out_path, legend=True,
                                   xlabel=r"$\alpha_1$: Capacity of Hospital 1 Agent",
                                   figsize=(8, 4.5),
                                   online_policies=["Logistic TS", "Logistic Greedy"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        logs = sorted((ROOT / "logs").glob("camelyon17_*.jsonl"))
        if not logs:
            print("No camelyon17 log files found.")
            sys.exit(1)
        log_path = logs[-1]

    out = make_camelyon17_plot(log_path)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
