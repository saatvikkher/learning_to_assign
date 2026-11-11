"""
Experiment: Camelyon17 — Hospital model routing

Two logistic models, each trained on patches from a different hospital
(WILDS Camelyon17 benchmark), are routed using image appearance features.
Context is 1D PCA of color/staining statistics plus a bias term.

Usage:
    python -m experiments.camelyon17 [options]

    --analyze         Train all 5 hospital models, print pair analysis, exit
    --hospital-a      Center index for weaker agent        (default: auto)
    --hospital-b      Center index for stronger agent      (default: auto)
    --ood-center      Center to include as OOD in test pool (default: auto)
    --n-runs          Number of runs per capacity          (default: 100)
    --capacities      Comma-separated weaker-agent capacity fractions
                                                           (default: 0.2,0.4,0.5,0.6,0.8)
    --sim-samples     Max patches in the experiment pool   (default: 2000)
    --eta             Lagrangian step size                 (default: 0.5)
    --seed            Base random seed                     (default: 42)
    --log-dir         Directory for JSONL logs             (default: logs/)
    --plot            Generate plot from most recent log and exit
    --log-file        Specific log file to plot (used with --plot)
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import run_online, run_offline, train_offline_models
from src.plotting import make_plot

DATA_ROOT = ROOT / "data" / "camelyon17_v1.0"
METADATA_PATH = DATA_ROOT / "metadata.csv"
FEATURE_CACHE = DATA_ROOT / "features_cache.npz"

# Per-center train sample cap (balanced tumor/non-tumor)
TRAIN_SAMPLES_PER_CENTER = 15_000


# ---------------------------------------------------------------------------
# Image path construction
# ---------------------------------------------------------------------------

def build_image_path(patient: int, node: int, x: int, y: int) -> Path:
    folder = f"patient_{patient:03d}_node_{node}"
    fname = f"patch_patient_{patient:03d}_node_{node}_x_{x}_y_{y}.png"
    return DATA_ROOT / "patches" / folder / fname


# ---------------------------------------------------------------------------
# Feature extraction — 21 features per 96×96 RGB patch
# ---------------------------------------------------------------------------

def _extract_one(args) -> tuple[int, np.ndarray | None]:
    """Worker: (row_idx, patient, node, x, y) → (row_idx, features)."""
    from PIL import Image
    row_idx, patient, node, x, y = args
    path = build_image_path(patient, node, x, y)
    try:
        arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    except Exception:
        return row_idx, None

    feats = []
    for c in range(3):
        ch = arr[:, :, c].ravel()
        feats += [
            ch.mean(),
            ch.std(),
            np.percentile(ch, 10),
            np.percentile(ch, 90),
            ch.max(),
        ]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    # Whiteness: fraction of pixels that look like background (near white)
    feats.append(((r > 220) & (g > 220) & (b > 220)).mean())
    # Hematoxylin proxy: blue–red (purpleness typical of nuclei staining)
    feats.append(float((b - r).mean()))
    # Eosin proxy: red–blue (pinkness of cytoplasm)
    feats.append(float((r - b).mean()))
    # Mean intensity
    feats.append(float(arr.mean()))
    # Colour contrast: std of grayscale
    feats.append(float(arr.mean(axis=2).std()))
    # Green dominance: often differs between staining protocols
    feats.append(float(g.mean()))

    return row_idx, np.array(feats, dtype=np.float32)


def extract_features(df: pd.DataFrame, n_workers: int = 8, desc: str = "") -> np.ndarray:
    """Extract 21-dim feature vectors for every row in df. Returns (n, 21) array."""
    args = [
        (idx, int(row.patient), int(row.node), int(row.x_coord), int(row.y_coord))
        for idx, row in df.iterrows()
    ]
    n = len(args)
    features = np.zeros((n, 21), dtype=np.float32)
    idx_map = {a[0]: i for i, a in enumerate(args)}  # row_idx → position in features

    print(f"  Extracting features for {n} images{' (' + desc + ')' if desc else ''}...")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_extract_one, a): a[0] for a in args}
        for fut in as_completed(futs):
            row_idx, feat = fut.result()
            if feat is not None:
                features[idx_map[row_idx]] = feat
            done += 1
            if done % 5000 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"    {done}/{n}  ({rate:.0f} img/s)", end="\r")
    print(f"    Done in {time.time()-t0:.1f}s          ")
    return features


# ---------------------------------------------------------------------------
# Feature cache: store features indexed by metadata row index
# ---------------------------------------------------------------------------

def load_or_build_feature_cache(df: pd.DataFrame, n_workers: int = 8) -> np.ndarray:
    """
    Returns features array aligned with df.index (shape = (len(df), 21)).
    Builds and saves a cache on first call; subsequent calls reload instantly.
    """
    if FEATURE_CACHE.exists():
        print(f"  Loading feature cache from {FEATURE_CACHE}...")
        cache = np.load(FEATURE_CACHE)
        cached_indices = cache["indices"]
        cached_features = cache["features"]
        # Build lookup: row_index → feature
        idx2feat = {int(idx): cached_features[i] for i, idx in enumerate(cached_indices)}
        missing = [idx for idx in df.index if idx not in idx2feat]
        if not missing:
            return np.stack([idx2feat[int(idx)] for idx in df.index])
        print(f"  Cache missing {len(missing)} entries — extracting...")
        missing_df = df.loc[missing]
        new_feats = extract_features(missing_df, n_workers=n_workers, desc="incremental")
        for i, idx in enumerate(missing):
            idx2feat[int(idx)] = new_feats[i]
        # Re-save full cache
        all_indices = np.array(list(idx2feat.keys()), dtype=np.int64)
        all_features = np.stack([idx2feat[int(i)] for i in all_indices])
        np.savez_compressed(FEATURE_CACHE, indices=all_indices, features=all_features)
        return np.stack([idx2feat[int(idx)] for idx in df.index])
    else:
        print(f"  No feature cache found — extracting all {len(df)} images...")
        features = extract_features(df, n_workers=n_workers, desc="full dataset")
        np.savez_compressed(
            FEATURE_CACHE,
            indices=np.array(df.index, dtype=np.int64),
            features=features,
        )
        print(f"  Feature cache saved to {FEATURE_CACHE}")
        return features


# ---------------------------------------------------------------------------
# Per-hospital model training
# ---------------------------------------------------------------------------

def train_hospital_models(
    df: pd.DataFrame,
    all_features: np.ndarray,
    seed: int = 42,
) -> dict[int, tuple[LogisticRegression, StandardScaler]]:
    """
    Train one logistic regression (on image features) per center using split=0 data.
    Returns {center: (model, scaler)}.
    """
    models = {}
    rng = np.random.default_rng(seed)
    for center in sorted(df["center"].unique()):
        mask = (df["center"] == center) & (df["split"] == 0)
        center_df = df[mask]
        center_feats = all_features[mask]
        labels = center_df["tumor"].values

        # Stratified subsample if too large
        n = len(labels)
        if n > TRAIN_SAMPLES_PER_CENTER:
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            half = TRAIN_SAMPLES_PER_CENTER // 2
            pos_sel = rng.choice(pos_idx, min(half, len(pos_idx)), replace=False)
            neg_sel = rng.choice(neg_idx, min(half, len(neg_idx)), replace=False)
            sel = np.concatenate([pos_sel, neg_sel])
            center_feats = center_feats[sel]
            labels = labels[sel]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(center_feats)
        model = LogisticRegression(
            solver="lbfgs", max_iter=1000, C=1.0, random_state=seed
        )
        model.fit(X_train, labels)

        train_acc = model.score(X_train, labels)
        print(
            f"  Center {center}: n_train={len(labels)}  "
            f"train_acc={train_acc:.4f}"
        )
        models[center] = (model, scaler)
    return models


def predict_with_model(
    model_scaler: tuple[LogisticRegression, StandardScaler],
    features: np.ndarray,
) -> np.ndarray:
    """Return predicted tumor labels (0 or 1)."""
    model, scaler = model_scaler
    return model.predict(scaler.transform(features))


# ---------------------------------------------------------------------------
# Pair analysis: find hospitals with complementary expertise
# ---------------------------------------------------------------------------

def analyze_pairs(
    df: pd.DataFrame,
    all_features: np.ndarray,
    models: dict,
    seed: int = 42,
) -> list[dict]:
    """
    For every pair of centers, evaluate on pooled split=1 data and compute:
      - accuracy of each model
      - complementarity: fraction of images where each model wins
      - separability: AUROC of a 1D PCA predicting which model wins
    Returns list of pair dicts sorted by complementarity score.
    """
    centers = sorted(models.keys())
    results = []

    for i, ca in enumerate(centers):
        for cb in centers[i + 1:]:
            # Pooled test data from both centers
            mask = (df["center"].isin([ca, cb])) & (df["split"] == 1)
            sub_df = df[mask]
            sub_feats = all_features[mask]
            labels = sub_df["tumor"].values

            pred_a = predict_with_model(models[ca], sub_feats)
            pred_b = predict_with_model(models[cb], sub_feats)

            correct_a = (pred_a == labels).astype(int)
            correct_b = (pred_b == labels).astype(int)

            acc_a = correct_a.mean()
            acc_b = correct_b.mean()

            # Disagreement set: one right, one wrong
            a_wins = (correct_a == 1) & (correct_b == 0)
            b_wins = (correct_b == 1) & (correct_a == 0)
            n_disagree = a_wins.sum() + b_wins.sum()

            if n_disagree < 50:
                sep_auroc = 0.5
            else:
                # Build a binary label: 1 if model_b wins, 0 if model_a wins
                dis_mask = a_wins | b_wins
                dis_feats = sub_feats[dis_mask]
                dis_label = b_wins[dis_mask].astype(int)

                # 1D PCA of features as context for separability check
                pca = PCA(n_components=1)
                ctx = pca.fit_transform(StandardScaler().fit_transform(dis_feats))
                try:
                    sep_auroc = roc_auc_score(dis_label, ctx[:, 0])
                    if sep_auroc < 0.5:
                        sep_auroc = 1.0 - sep_auroc  # directional symmetry
                except Exception:
                    sep_auroc = 0.5

            # Complementarity: how balanced are the wins?
            frac_a = a_wins.sum() / max(n_disagree, 1)
            frac_b = b_wins.sum() / max(n_disagree, 1)
            balance = min(frac_a, frac_b) / 0.5  # 1.0 = perfect balance

            # Overall score: balance × separability × min_accuracy
            min_acc = min(acc_a, acc_b)
            score = balance * (sep_auroc - 0.5) * 2 * max(min_acc - 0.5, 0.0)

            results.append(
                dict(
                    ca=ca,
                    cb=cb,
                    acc_a=acc_a,
                    acc_b=acc_b,
                    n_test=len(labels),
                    n_disagree=int(n_disagree),
                    frac_a_wins=float(frac_a),
                    frac_b_wins=float(frac_b),
                    sep_auroc=float(sep_auroc),
                    balance=float(balance),
                    score=float(score),
                )
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def print_analysis(results: list[dict]) -> None:
    print("\n=== Hospital Pair Analysis ===")
    header = (
        f"{'Pair':>8}  {'Acc_A':>6}  {'Acc_B':>6}  "
        f"{'N_test':>7}  {'Disagr%':>7}  "
        f"{'FracA':>6}  {'FracB':>6}  "
        f"{'AUROC':>6}  {'Score':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"  ({r['ca']},{r['cb']})  "
            f"{r['acc_a']:.4f}  {r['acc_b']:.4f}  "
            f"{r['n_test']:7d}  "
            f"{100*r['n_disagree']/r['n_test']:6.1f}%  "
            f"{r['frac_a_wins']:.4f}  {r['frac_b_wins']:.4f}  "
            f"{r['sep_auroc']:.4f}  {r['score']:.5f}"
        )
    print()
    best = results[0]
    weaker_center = best["ca"] if best["acc_a"] <= best["acc_b"] else best["cb"]
    stronger_center = best["cb"] if best["acc_a"] <= best["acc_b"] else best["ca"]
    print(f"Best pair: centers ({best['ca']}, {best['cb']})")
    print(f"  Weaker  (Agent 0): center {weaker_center}  acc={min(best['acc_a'], best['acc_b']):.4f}")
    print(f"  Stronger (Agent 1): center {stronger_center}  acc={max(best['acc_a'], best['acc_b']):.4f}")
    print(f"  Separability AUROC: {best['sep_auroc']:.4f}")
    print(f"  Disagreement rate: {100*best['n_disagree']/best['n_test']:.1f}%")


# ---------------------------------------------------------------------------
# Experiment data preparation
# ---------------------------------------------------------------------------

def prepare_experiment_data(
    df: pd.DataFrame,
    all_features: np.ndarray,
    center_a: int,
    center_b: int,
    models: dict,
    ood_center: int | None,
    sim_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float, float, str, str]:
    """
    Returns (X, rewards_matrix, acc0, acc1, name0, name1) following conventions.
    Agent 0 is always the weaker model.
    X shape: (n, 2) — 1D PCA of features + bias.
    rewards_matrix shape: (n, 2) — binary correct/incorrect for each model.
    """
    rng = np.random.default_rng(seed)
    centers_to_include = [center_a, center_b]
    if ood_center is not None:
        centers_to_include.append(ood_center)

    mask = (df["center"].isin(centers_to_include)) & (df["split"] == 1)
    pool_df = df[mask].copy()
    pool_feats = all_features[mask]
    labels = pool_df["tumor"].values

    # Subsample if pool is too large
    if len(pool_df) > sim_samples:
        sel = rng.choice(len(pool_df), sim_samples, replace=False)
        pool_feats = pool_feats[sel]
        labels = labels[sel]

    pred_a = predict_with_model(models[center_a], pool_feats)
    pred_b = predict_with_model(models[center_b], pool_feats)

    correct_a = (pred_a == labels).astype(float)
    correct_b = (pred_b == labels).astype(float)

    acc_a = correct_a.mean()
    acc_b = correct_b.mean()

    # Agent 0 = weaker (lower accuracy)
    if acc_a <= acc_b:
        rewards_matrix = np.column_stack([correct_a, correct_b])
        name0, name1 = f"hospital_{center_a}", f"hospital_{center_b}"
    else:
        rewards_matrix = np.column_stack([correct_b, correct_a])
        acc_a, acc_b = acc_b, acc_a
        name0, name1 = f"hospital_{center_b}", f"hospital_{center_a}"

    # 1D PCA of image features + bias term
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(pool_feats)
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(feats_scaled)
    X = np.column_stack([X_pca, np.ones(len(X_pca))])

    return X, rewards_matrix, acc_a, acc_b, name0, name1


# ---------------------------------------------------------------------------
# Log helpers (mirror mmlu.py)
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


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args: argparse.Namespace) -> Path:
    df = pd.read_csv(METADATA_PATH, index_col=0)
    print(f"Metadata: {len(df)} patches, {df['center'].nunique()} centers")

    # Feature extraction (uses full df for cache alignment)
    print("Loading/building feature cache...")
    all_features = load_or_build_feature_cache(df, n_workers=args.n_workers)

    # Train per-hospital models
    print("Training per-hospital models (split=0 data)...")
    models = train_hospital_models(df, all_features, seed=args.seed)

    # Pair analysis
    print("Analyzing hospital pairs...")
    pair_results = analyze_pairs(df, all_features, models, seed=args.seed)
    print_analysis(pair_results)

    if args.analyze:
        return None

    # Select hospital pair
    best = pair_results[0]
    center_a = args.hospital_a if args.hospital_a >= 0 else best["ca"]
    center_b = args.hospital_b if args.hospital_b >= 0 else best["cb"]

    # Auto-select OOD center: smallest center not in the chosen pair
    all_centers = sorted(df["center"].unique())
    ood_candidates = [c for c in all_centers if c not in (center_a, center_b)]
    if args.ood_center >= 0:
        ood_center = args.ood_center
    elif ood_candidates:
        # Pick the one with fewest test samples for diversity without overwhelming
        sizes = {c: ((df["center"] == c) & (df["split"] == 1)).sum() for c in ood_candidates}
        ood_center = min(sizes, key=sizes.get)
    else:
        ood_center = None

    print(f"\nExperiment setup:")
    print(f"  Center A: {center_a}  |  Center B: {center_b}  |  OOD: {ood_center}")

    # Prepare experiment data
    print("\nPreparing experiment pool (split=1 data)...")
    X, rewards_matrix, acc0, acc1, name0, name1 = prepare_experiment_data(
        df, all_features, center_a, center_b, models,
        ood_center=ood_center, sim_samples=args.sim_samples, seed=args.seed,
    )
    n = len(X)
    print(f"  Pool size: {n}")
    print(f"  Agent 0 (weaker):   {name0}  acc={acc0:.4f}  err={1-acc0:.4f}")
    print(f"  Agent 1 (stronger): {name1}  acc={acc1:.4f}  err={1-acc1:.4f}")

    capacities = [float(c) for c in args.capacities.split(",")]

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"camelyon17_{timestamp}.jsonl"
    log_path = log_dir / log_name
    print(f"Logging to {log_path}")

    meta = {
        "experiment": "camelyon17",
        "center_a": int(center_a),
        "center_b": int(center_b),
        "ood_center": int(ood_center) if ood_center is not None else None,
        "agent0": name0,
        "agent1": name1,
        "eta": args.eta,
        "seed": args.seed,
    }

    print("Training offline baselines...")
    offline_models = train_offline_models(
        X, rewards_matrix[:, 0], rewards_matrix[:, 1], seed=args.seed
    )
    base_perm = np.arange(n)
    unc_log_acc = run_offline(
        X, rewards_matrix, np.zeros(2), base_perm,
        "offline_logistic_unconstrained", offline_models, args.eta
    )
    unc_tree_acc = run_offline(
        X, rewards_matrix, np.zeros(2), base_perm,
        "offline_tree_unconstrained", offline_models, args.eta
    )
    print(f"  Unconstrained Logistic acc={unc_log_acc:.4f}  Tree acc={unc_tree_acc:.4f}")

    with open(log_path, "w") as f:
        log_boundary(f, 0.0, 1.0 - acc1, meta)
        log_boundary(f, 1.0, 1.0 - acc0, meta)
        log_boundary(f, -1.0, 1.0 - unc_log_acc, {**meta, "policy": "Unconstrained Logistic"})
        log_boundary(f, -1.0, 1.0 - unc_tree_acc, {**meta, "policy": "Unconstrained Tree"})

        total = len(capacities) * args.n_runs
        done = 0
        t0 = time.time()

        for run_idx in range(args.n_runs):
            perm_rng = np.random.default_rng([args.seed, run_idx, 0])
            perm = perm_rng.permutation(n)

            for cap in capacities:
                p_targets = np.array([cap, 1.0 - cap])
                policies: dict[str, float] = {}

                policies["Offline Logistic"] = run_offline(
                    X, rewards_matrix, p_targets, perm, "offline_logistic", offline_models, args.eta
                )
                policies["Offline Tree"] = run_offline(
                    X, rewards_matrix, p_targets, perm, "offline_tree", offline_models, args.eta
                )

                for algo_idx, (mode, label) in enumerate([
                    ("log_ts",      "Logistic TS"),
                    ("log_greedy",  "Logistic Greedy"),
                    ("tree_ts",     "Tree TS"),
                    ("tree_greedy", "Tree Greedy"),
                    ("random",      "Random"),
                ]):
                    algo_rng = np.random.default_rng([args.seed, run_idx, algo_idx + 1])
                    policies[label] = run_online(
                        X, rewards_matrix, p_targets, perm, mode, args.eta, rng=algo_rng
                    )

                log_run(f, cap, run_idx, policies, meta)

                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (total - done) / rate if rate > 0 else 0
                print(
                    f"  run {run_idx+1}/{args.n_runs}  cap={cap}  "
                    f"[{done}/{total} | {remaining/60:.1f} min left]",
                    end="\r",
                )

            print()

    print(f"\nDone. Log: {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Camelyon17 hospital routing experiment")
    parser.add_argument("--analyze", action="store_true", help="Train models, print pair analysis, exit")
    parser.add_argument("--hospital-a", type=int, default=-1, help="Center index for agent A (-1 = auto)")
    parser.add_argument("--hospital-b", type=int, default=-1, help="Center index for agent B (-1 = auto)")
    parser.add_argument("--ood-center", type=int, default=-1, help="OOD center index (-1 = auto)")
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--capacities", default="0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--sim-samples", type=int, default=2000)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--n-workers", type=int, default=8, help="Parallel workers for feature extraction")
    parser.add_argument("--plot", action="store_true", help="Plot from most recent log and exit")
    parser.add_argument("--log-file", default=None, help="Specific log file to plot (with --plot)")
    args = parser.parse_args()

    if args.plot:
        log_dir = Path(args.log_dir)
        if args.log_file is None:
            logs = sorted(log_dir.glob("camelyon17_*.jsonl"))
            if not logs:
                print("No camelyon17 log files found.")
                sys.exit(1)
            log_file = logs[-1]
        else:
            log_file = Path(args.log_file)
        out = make_plot(log_file, xlabel="Capacity (Agent 1 — stronger)")
        print(f"Plot saved to {out}")
        return

    log_path = run_experiment(args)
    if log_path is not None:
        print("Generating plot...")
        out = make_plot(log_path, xlabel="Capacity (Agent 1 — stronger)")
        print(f"Plot saved to {out}")


if __name__ == "__main__":
    main()
