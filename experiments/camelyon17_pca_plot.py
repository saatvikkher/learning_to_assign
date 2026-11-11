"""
PCA distribution + smoothed accuracy plot for the Camelyon17 experiment.
Uses all split=1 test patches (~31K) from hospitals 3, 4, and 1 (OOD).
Mirrors mmlu_pca_plot.py conventions.
Saves to plots/camelyon17_pca*.pdf
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_ROOT  = ROOT / "data" / "camelyon17_v1.0"
META_PATH  = DATA_ROOT / "metadata.csv"
CACHE_PATH = DATA_ROOT / "features_cache.npz"

CENTER_A   = 3
CENTER_B   = 4
CENTER_OOD = 1
SEED       = 42
TRAIN_CAP  = 15_000

COLOR_HIST_A   = "#8fbc8f"   # sage green  — hospital 3
COLOR_HIST_B   = "#b0a0c8"   # lavender    — hospital 4
COLOR_HIST_OOD = "#f0c080"   # warm amber  — hospital 1 (OOD)

COLOR_LINE_A   = "#2e7d32"   # dark green  — hospital 3 model
COLOR_LINE_B   = "#4527a0"   # deep purple — hospital 4 model


def kernel_smooth(x_obs, y_obs, x_eval, bw):
    w = np.exp(-0.5 * ((x_eval[:, None] - x_obs[None, :]) / bw) ** 2)
    w /= w.sum(axis=1, keepdims=True) + 1e-12
    return w @ y_obs


def train(df, idx2pos, feat_arr, center):
    rng = np.random.default_rng(SEED)
    sub = df[(df["center"] == center) & (df["split"] == 0)]
    X   = feat_arr[[idx2pos[int(i)] for i in sub.index]]
    y   = sub["tumor"].values
    if len(y) > TRAIN_CAP:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        half = TRAIN_CAP // 2
        sel  = np.concatenate([rng.choice(pos, min(half, len(pos)), replace=False),
                               rng.choice(neg, min(half, len(neg)), replace=False)])
        X, y = X[sel], y[sel]
    sc = StandardScaler()
    m  = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0, random_state=SEED)
    m.fit(sc.fit_transform(X), y)
    return m, sc


def main():
    print("Loading feature cache...")
    cache    = np.load(CACHE_PATH)
    idx2pos  = {int(i): k for k, i in enumerate(cache["indices"])}
    feat_arr = cache["features"]

    df = pd.read_csv(META_PATH, index_col=0)

    print("Training hospital models...")
    model_a, sc_a = train(df, idx2pos, feat_arr, CENTER_A)
    model_b, sc_b = train(df, idx2pos, feat_arr, CENTER_B)

    print("Building data (centers 3, 4, 1, split=1 — ~31K points)...")
    sub   = df[df["center"].isin([CENTER_A, CENTER_B, CENTER_OOD]) & (df["split"] == 1)]
    feats = feat_arr[[idx2pos[int(i)] for i in sub.index]]
    src   = sub["center"].values
    lbls  = sub["tumor"].values

    correct_a = (model_a.predict(sc_a.transform(feats)) == lbls).astype(float)
    correct_b = (model_b.predict(sc_b.transform(feats)) == lbls).astype(float)
    print(f"  Hospital {CENTER_A} model acc: {correct_a.mean():.4f}")
    print(f"  Hospital {CENTER_B} model acc: {correct_b.mean():.4f}")

    print("Fitting 1D PCA...")
    x1d = PCA(n_components=1, random_state=SEED).fit_transform(
              StandardScaler().fit_transform(feats)).ravel()

    x_eval = np.linspace(x1d.min(), x1d.max(), 300)
    bw     = (x1d.max() - x1d.min()) / 14.0
    bins   = np.linspace(x1d.min(), x1d.max(), 42)

    smooth_a = kernel_smooth(x1d, correct_a, x_eval, bw)
    smooth_b = kernel_smooth(x1d, correct_b, x_eval, bw)
    delta    = smooth_b - smooth_a

    # ── precompute histogram label positions (shared by all figures) ──────────
    x_range      = bins[-1] - bins[0]
    close_thresh = x_range / 7.0

    hist_info = []
    for c, col, lbl, lcol in [
        (CENTER_A,   COLOR_HIST_A,   "Hospital 1",         "#4a8a4a"),
        (CENTER_B,   COLOR_HIST_B,   "Hospital 2",         "#7060a8"),
        (CENTER_OOD, COLOR_HIST_OOD, "Hospital 3 (OOD)",   "#b07820"),
    ]:
        counts, _ = np.histogram(x1d[src == c], bins=bins)
        bi = int(np.argmax(counts))
        peak_x = (bins[bi] + bins[bi + 1]) / 2
        hist_info.append((peak_x, int(counts[bi]), col, lbl, lcol))

    max_count = max(h[1] for h in hist_info)
    headroom  = max_count * 0.07
    step      = max_count * 0.14

    order    = np.argsort([h[0] for h in hist_info])
    placed   = []
    label_ys = [0.0] * len(hist_info)
    for idx in order:
        px, pc = hist_info[idx][0], hist_info[idx][1]
        y = pc + headroom
        for lx, ly in placed:
            if abs(px - lx) < close_thresh:
                y = max(y, ly + step)
        label_ys[idx] = y
        placed.append((px, y))

    # Hospital 2 label sits to the right at x≈5, level with Hospital 1 label
    label_ys[1] = label_ys[0]
    label_xs    = [hi[0] for hi in hist_info]
    label_xs[1] = 5.0

    label_coords = [
        (label_xs[0] + 0.2, label_ys[0]),
        (label_xs[1] + 1,   label_ys[1]),
        (label_xs[2] + 3.5, label_ys[2] - 500),
    ]

    def make_figure(ann_frac, xlim_left, out_path,
                    show_ood=True, show_curves=True, show_delta=True):
        fig, ax_acc = plt.subplots(figsize=(8, 5))
        ax_hist = ax_acc.twinx()

        hist_centers = [(CENTER_A, COLOR_HIST_A), (CENTER_B, COLOR_HIST_B)]
        if show_ood:
            hist_centers.append((CENTER_OOD, COLOR_HIST_OOD))
        for c, col in hist_centers:
            ax_hist.hist(x1d[src == c], bins=bins, color=col, alpha=0.40)

        ax_hist.set_ylabel("Count", fontsize=17)
        ax_hist.tick_params(axis="y", labelcolor="#777777", labelsize=15)
        ax_hist.spines[["top"]].set_visible(True)
        ax_hist.set_ylim(0, max_count * 1.55)
        ax_hist.set_zorder(1)
        ax_acc.set_zorder(2)
        ax_acc.patch.set_visible(False)

        if show_curves:
            ax_acc.plot(x_eval, smooth_a, color=COLOR_LINE_A, linewidth=2.2)
            ax_acc.plot(x_eval, smooth_b, color=COLOR_LINE_B, linewidth=2.2)

        if show_delta:
            ax_acc.plot(x_eval, delta, color="#aaaaaa", linewidth=2,
                        linestyle="--", zorder=3)

        ax_acc.tick_params(axis="both", labelsize=15)
        ax_acc.set_xlabel(r"Context $x$: 1D PCA", fontsize=17)
        ax_acc.set_ylim(-0.55, 1.05)
        ax_acc.grid(alpha=0.1, linewidth=0.7)
        ax_acc.spines[["top"]].set_visible(True)
        ax_acc.spines[["right"]].set_visible(False)

        if show_curves:
            ax_acc.set_ylabel("Accuracy", fontsize=17)
            ax_acc.tick_params(axis="y", labelcolor="black", color="black")
        else:
            # Keep label and ticks invisible so tight_layout reserves identical space
            ax_acc.set_ylabel("Accuracy", fontsize=17, color="none")
            ax_acc.tick_params(axis="y", labelcolor="none", color="none")

        if xlim_left is not None:
            ax_acc.set_xlim(xlim_left, x1d.max())
            ax_hist.set_xlim(xlim_left, x1d.max())

        if show_curves:
            ann_idx  = int(ann_frac * len(x_eval))
            ann_x    = x_eval[ann_idx]
            y_a, y_b = smooth_a[ann_idx], smooth_b[ann_idx]
            gap      = abs(y_b - y_a)
            offset   = max(0.09, 0.04 + (0.12 - gap) / 2) if gap < 0.12 else 0.09
            ax_acc.text(ann_x, y_a + offset - 0.22, "Hospital 1 Agent",
                        fontsize=11, color=COLOR_LINE_A, ha="right", va="bottom")
            ax_acc.text(ann_x, y_b - offset + 0.2, "Hospital 2 Agent",
                        fontsize=11, color=COLOR_LINE_B, ha="right", va="top")

        if show_delta:
            x_lo = xlim_left if xlim_left is not None else x1d.min()
            visible = (x_eval >= x_lo) & (x_eval <= x1d.max())
            delta_peak = int(np.where(visible)[0][0]) + int(np.argmax(np.abs(delta[visible])))
            ax_acc.text(x_eval[delta_peak] - 0.2, delta[delta_peak] - 0.1,
                        r"$\mathbf{\Delta(x)}$",
                        fontsize=12, color="#999999", ha="right", fontweight="bold")

        n_labels = 3 if show_ood else 2
        for idx in range(n_labels):
            col, lbl, lcol = hist_info[idx][2], hist_info[idx][3], hist_info[idx][4]
            x, y = label_coords[idx]
            ax_hist.text(x, y, lbl, color=lcol, fontsize=11,
                         ha="center", fontweight="bold", zorder=1)

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {out}")

    PLOTS = ROOT / "plots"
    kw = dict(ann_frac=0.82, xlim_left=-12)
    make_figure(**kw, show_ood=False, show_curves=False, show_delta=False,
                out_path=PLOTS / "camelyon17_pca_1_hist.pdf")
    make_figure(**kw, show_ood=False, show_curves=True,  show_delta=False,
                out_path=PLOTS / "camelyon17_pca_2_curves.pdf")
    make_figure(**kw, show_ood=False, show_curves=True,  show_delta=True,
                out_path=PLOTS / "camelyon17_pca_3_delta.pdf")
    make_figure(**kw, show_ood=True,  show_curves=True,  show_delta=True,
                out_path=PLOTS / "camelyon17_pca.pdf")


if __name__ == "__main__":
    main()
