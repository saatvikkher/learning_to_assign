"""
PCA distribution + smoothed accuracy plot for the MMLU experiment.
Saves to plots/mmlu_pca.pdf
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "mmlu_scored_progress.csv"
MODEL_A = "cohere/small-20220720"
MODEL_B = "meta/llama-13b"
CAT1 = "college_chemistry"
CAT2 = "us_foreign_policy"
EMBEDDER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Histogram: muted greens so they don't clash with the accuracy line colors
COLOR_CAT1_HIST = "#8fbc8f"   # muted sage green
COLOR_CAT2_HIST = "#b0a0c8"   # muted lavender

# Accuracy lines: warm red vs dark teal — clearly distinct from the hist colors
COLOR_MODEL_A = "#c0392b"     # Cohere Small — warm red
COLOR_MODEL_B = "#1a6b8a"     # LLaMA-13B   — dark teal


def kernel_smooth(x_obs, y_obs, x_eval, bandwidth):
    w = np.exp(-0.5 * ((x_eval[:, None] - x_obs[None, :]) / bandwidth) ** 2)
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    return w @ y_obs


def main():
    from sentence_transformers import SentenceTransformer
    import torch

    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates(subset=["prompt", "category", "model"])

    exp_df = df[
        df["category"].isin([CAT1, CAT2]) & df["model"].isin([MODEL_A, MODEL_B])
    ]
    pivot = exp_df.pivot(index="prompt", columns="model", values="is_correct").dropna()
    prompts = pivot.index.tolist()

    cat_map = (
        exp_df.drop_duplicates("prompt")
        .set_index("prompt")
        .loc[prompts, "category"]
        .values
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding {len(prompts)} prompts on {device}...")
    embedder = SentenceTransformer(EMBEDDER_MODEL, device=device)
    embeddings = embedder.encode(prompts, batch_size=64, show_progress_bar=False)
    embedder.cpu()
    del embedder

    pca = PCA(n_components=1)
    X1d = pca.fit_transform(embeddings).ravel()

    rewards_a = pivot[MODEL_A].values.astype(float)
    rewards_b = pivot[MODEL_B].values.astype(float)

    x_eval = np.linspace(X1d.min(), X1d.max(), 200)
    bw = (X1d.max() - X1d.min()) / 10.0
    smooth_a = kernel_smooth(X1d, rewards_a, x_eval, bw)
    smooth_b = kernel_smooth(X1d, rewards_b, x_eval, bw)
    delta = smooth_b - smooth_a   # positive where LLaMA leads

    # ── single figure, two y-axes ───────────────────────────────────────────
    fig, ax_acc = plt.subplots(figsize=(9.5, 3.6))
    ax_hist = ax_acc.twinx()

    # ── histogram (back layer) ───────────────────────────────────────────────
    bins = np.linspace(X1d.min(), X1d.max(), 28)
    ax_hist.hist(X1d[cat_map == CAT1], bins=bins, color=COLOR_CAT1_HIST, alpha=0.45)
    ax_hist.hist(X1d[cat_map == CAT2], bins=bins, color=COLOR_CAT2_HIST, alpha=0.45)
    ax_hist.set_ylabel("Count", fontsize=17)
    ax_hist.tick_params(axis="y", labelcolor="#777777", labelsize=15)
    ax_hist.spines[["top"]].set_visible(False)
    # push hist behind the accuracy lines
    ax_hist.set_zorder(1)
    ax_acc.set_zorder(2)
    ax_acc.patch.set_visible(False)

    # ── accuracy lines (front layer) ─────────────────────────────────────────
    ax_acc.plot(x_eval, smooth_a, color=COLOR_MODEL_A, linewidth=2.2)
    ax_acc.plot(x_eval, smooth_b, color=COLOR_MODEL_B, linewidth=2.2)
    ax_acc.plot(x_eval, delta,    color="#aaaaaa", linewidth=2,
                linestyle="--", zorder=3)
    ax_acc.axhline(0, color="#cccccc", linewidth=0.6, zorder=1)

    ax_acc.tick_params(axis="both", labelsize=15)
    ax_acc.set_xlabel(r"Context $x$: 1D PCA", fontsize=17)
    ax_acc.set_ylabel("Accuracy", fontsize=17)
    ax_acc.set_ylim(-0.55, 1.05)
    ax_acc.grid(alpha=0.2, linewidth=0.7)
    ax_acc.spines[["top", "right"]].set_visible(False)

    # ── line annotations ─────────────────────────────────────────────────────
    # place labels where lines have clearly separated (~60% of x range)
    ann_idx = int(0.62 * len(x_eval))
    ann_x = x_eval[ann_idx]

    # "Cohere Small" below the red curve
    ax_acc.text(
        ann_x + 0.03, smooth_a[ann_idx] - 0.09, "Cohere Small",
        fontsize=12, color=COLOR_MODEL_A, ha="right", va="top",
    )
    # "LLaMA-13B" above the blue curve, right edge near "F" of Foreign Policy
    ax_acc.text(
        ann_x, smooth_b[ann_idx] + 0.09, "LLaMA-13B",
        fontsize=12, color=COLOR_MODEL_B, ha="right", va="bottom",
    )
    # delta label near its peak
    delta_peak_idx = int(np.argmax(delta))
    ax_acc.text(
        x_eval[delta_peak_idx] + 0.05, delta[delta_peak_idx] + 0.05, r"$\mathbf{\Delta(x)}$",
        fontsize=12, color="#999999", ha="right", fontweight="bold"
    )

    # ── category labels low on the chart, out of the way ─────────────────────
    cat1_peak = X1d[cat_map == CAT1].mean()
    cat2_peak = X1d[cat_map == CAT2].mean()
    ax_acc.text(
        cat1_peak, -0.40, "Chemistry",
            color="#7fa77f", fontsize=12, ha="center", fontweight="bold",
    )
    ax_acc.text(
        cat2_peak, -0.40, "Foreign Policy",
            color="#9b89b6", fontsize=12, ha="center", fontweight="bold",
    )

    out = ROOT / "plots" / "mmlu_pca.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
