"""
Results plot for MMLU.
Computes offline unconstrained baselines (trained on the same test pool as the
online experiment) and renders the styled results plot.

Usage:
    python -m experiments.mmlu_results_plot [--log-file logs/mmlu_*.jsonl]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.plotting import load_log, make_styled_results_plot

DATA_PATH = ROOT / "data" / "mmlu_scored_progress.csv"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def compute_offline_unconstrained(log_path: Path) -> tuple[dict, list]:
    """
    Reconstruct X and rewards from log metadata, train one logistic and one RF
    per agent's reward (same hyperparams as online), route to argmax, and
    return (error_rates_dict, assignment_ratios).
    """
    import json, pandas as pd
    from sentence_transformers import SentenceTransformer
    import torch

    with open(log_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("run", -1) >= 0:
                seed    = rec["seed"]
                model_a = rec["model_a"]
                model_b = rec["model_b"]
                cat1    = rec["cat1"]
                cat2    = rec["cat2"]
                break

    df = pd.read_csv(DATA_PATH).drop_duplicates(subset=["prompt", "category", "model"])
    exp_df = df[df["category"].isin([cat1, cat2]) & df["model"].isin([model_a, model_b])]
    pivot  = exp_df.pivot(index="prompt", columns="model", values="is_correct").dropna()
    prompts  = pivot.index.tolist()
    r_a = pivot[model_a].values.astype(float)
    r_b = pivot[model_b].values.astype(float)
    if r_a.mean() <= r_b.mean():
        rewards = np.column_stack([r_a, r_b])
    else:
        rewards = np.column_stack([r_b, r_a])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Embedding {len(prompts)} prompts on {device}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL, device=device)
    emb = embedder.encode(prompts, batch_size=64, show_progress_bar=False)
    del embedder

    X_pca = PCA(n_components=1, random_state=seed).fit_transform(emb).ravel()
    X = np.column_stack([X_pca, np.ones(len(X_pca))])
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


def make_mmlu_plot(log_path: Path) -> Path:
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
        logs = sorted((ROOT / "logs").glob("mmlu_*.jsonl"))
        if not logs:
            print("No mmlu log files found.")
            sys.exit(1)
        log_path = logs[-1]

    out = make_mmlu_plot(log_path)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
