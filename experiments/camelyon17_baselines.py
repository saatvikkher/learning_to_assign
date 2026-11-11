"""
Camelyon17 baseline comparisons (hospital 3 vs hospital 4, OOD hospital 1; paper stream).

Setups (--setup):
  baselines  Full paper stream (T = 2000). Online plug-in quantile threshold (Logistic, Tree),
             Dual Mirror Descent (Euclidean / Entropic x Logistic / Tree; arXiv:2011.10124),
             and our queue policy (Logistic / Tree Greedy) at eta = 0.5 and eta = sqrt(A/T).
  eta0       Full paper stream. Our policies with eta = 0 (no capacity pressure):
             Logistic TS vs Greedy, Tree TS vs Greedy. Results do not depend on capacity.
  split      50/50 split of the 2000-patch pool, stratified by hospital. Offline routers are
             fit on one half; the stream is the other half (T = 1000). Online policies
             (Logistic/Tree x TS/Greedy, eta = 0.5) + Random, and offline routers with the
             virtual queue (eta = 0.5) plus their unconstrained versions.

Seeds follow CLAUDE.md: perm = default_rng([seed, run, 0]); each policy gets a fresh generator
default_rng([seed, run, k]). Our online policies use the paper's k (algo_idx + 1), so the
eta = 0.5 runs reproduce the paper log, and variants with other eta share those
random numbers.

Usage:
    OMP_NUM_THREADS=1 python -m experiments.camelyon17_baselines --setup baselines --workers 40
    python -m experiments.camelyon17_baselines --plot logs/camelyon17_baselines_*.jsonl
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

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation import run_online, train_offline_models
from src.baselines import run_online_plugin, run_dmd

ONLINE_K = {"log_ts": 1, "log_greedy": 2, "tree_ts": 3, "tree_greedy": 4, "random": 5}
ONLINE_LABEL = {"log_ts": "Logistic TS", "log_greedy": "Logistic Greedy",
                "tree_ts": "Tree TS", "tree_greedy": "Tree Greedy", "random": "Random"}
BURN_IN = 100


def policy_specs(setup: str, T: int) -> dict:
    eta_sqrt = float(np.sqrt(2 / T))
    if setup == "baselines":
        specs = {
            "Plug-in Logistic": ("plugin", "logistic"),
            "Plug-in Tree": ("plugin", "tree"),
            "DMD Euclidean Logistic": ("dmd", "logistic", "euclidean"),
            "DMD Euclidean Tree": ("dmd", "tree", "euclidean"),
            "DMD Entropic Logistic": ("dmd", "logistic", "entropic"),
            "DMD Entropic Tree": ("dmd", "tree", "entropic"),
        }
        for mode in ("log_greedy", "tree_greedy"):
            specs[f"{ONLINE_LABEL[mode]} (eta=0.5)"] = ("online", mode, 0.5)
            specs[f"{ONLINE_LABEL[mode]} (eta=sqrt(A/T))"] = ("online", mode, eta_sqrt)
        return specs
    if setup == "eta0":
        return {f"{ONLINE_LABEL[m]} (eta=0)": ("online", m, 0.0)
                for m in ("log_ts", "log_greedy", "tree_ts", "tree_greedy")}
    specs = {ONLINE_LABEL[m]: ("online", m, 0.5) for m in ONLINE_K}
    specs["Offline Logistic"] = ("offline", "log", 0.5)
    specs["Offline Tree"] = ("offline", "tree", 0.5)
    return specs


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_stream(args):
    """Paper stream (X, R) plus each patch's hospital, agent 0 = weaker."""
    import experiments.camelyon17 as cam

    df = pd.read_csv(cam.METADATA_PATH, index_col=0)
    feats = cam.load_or_build_feature_cache(df)
    models = cam.train_hospital_models(df, feats, seed=args.seed)
    X, R, acc0, acc1, name0, name1 = cam.prepare_experiment_data(
        df, feats, args.hospital_a, args.hospital_b, models, args.ood_center, args.sim_samples, args.seed)
    # Same subsample as prepare_experiment_data, to recover the hospital of each patch
    mask = (df["center"].isin([args.hospital_a, args.hospital_b, args.ood_center])) & (df["split"] == 1)
    centers = df["center"].values[mask.values]
    if len(centers) > args.sim_samples:
        centers = centers[np.random.default_rng(args.seed).choice(len(centers), args.sim_samples, replace=False)]
    return X, R, centers, name0, name1


def stratified_half(groups, rng):
    is_test = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        idx = rng.permutation(np.where(groups == g)[0])
        is_test[idx[: len(idx) // 2]] = True
    return is_test


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ENV: dict = {}


def offline_queue(S, R, p, perm, eta):
    """Fixed router + virtual queue (same rule/tie-break as src.simulation.run_offline)."""
    S, Rs = S[perm], R[perm]
    Q = np.zeros(2)
    actions = np.zeros(len(S), dtype=int)
    for t in range(len(S)):
        a = 0 if S[t, 0] - eta * Q[0] >= S[t, 1] - eta * Q[1] else 1
        actions[t] = a
        z = np.zeros(2)
        z[a] = 1.0
        Q = np.maximum(Q + z - p, 0.0)
    return float(Rs[np.arange(len(S)), actions].mean()), actions


def _task(run_idx: int, cap: float, label: str):
    X, R, seed = _ENV["X"], _ENV["R"], _ENV["seed"]
    spec = _ENV["specs"][label]
    perm = np.random.default_rng([seed, run_idx, 0]).permutation(len(X))
    p = np.array([cap, 1.0 - cap])
    extra = {}
    if spec[0] == "online":
        _, mode, eta = spec
        rng = np.random.default_rng([seed, run_idx, ONLINE_K[mode]])
        avg, acts = run_online(X, R, p, perm, mode, eta, rng=rng, return_actions=True)
    elif spec[0] == "plugin":
        k = 11 if spec[1] == "logistic" else 12
        avg, acts = run_online_plugin(X, R, cap, perm, spec[1],
                                      np.random.default_rng([seed, run_idx, k]),
                                      np.random.default_rng([seed, run_idx, k + 10]))
    elif spec[0] == "dmd":
        k = {"logistic": 31, "tree": 32}[spec[1]] + (2 if spec[2] == "entropic" else 0)
        avg, acts, extra = run_dmd(X, R, cap, perm, spec[1], spec[2], np.random.default_rng([seed, run_idx, k]))
    else:
        avg, acts = offline_queue(_ENV["S_" + spec[1]], R, p, perm, spec[2])
    share = np.cumsum(acts == 0) / np.arange(1, len(acts) + 1)
    track = {"share0": float(share[-1]), "max_dev": float(np.abs(share[BURN_IN - 1:] - cap).max()), **extra}
    return run_idx, cap, label, float(avg), track, acts.astype(np.int8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args) -> Path:
    X, R, centers, name0, name1 = load_stream(args)
    meta_extra = {}
    if args.setup == "split":
        te = stratified_half(centers, np.random.default_rng([args.seed, 3000]))
        X_tr, R_tr = X[~te], R[~te]
        X, R, c_te = X[te], R[te], centers[te]
        models = train_offline_models(X_tr, R_tr[:, 0], R_tr[:, 1], seed=args.seed)
        log1, log2, tree1, tree2 = models
        _ENV["S_log"] = np.column_stack([log1.predict_proba(X)[:, 1], log2.predict_proba(X)[:, 1]])
        _ENV["S_tree"] = np.column_stack([tree1.predict(X), tree2.predict(X)])
        # Agent order is kept as in the paper stream (agent 0 = hospital 3) so capacities mean
        # the same thing across setups; on this half the two hospitals are near-tied
        # (0.699 vs 0.700), so the usual "agent 0 = weaker" swap would flip it.
        print(f"  split: {int((~te).sum())} router-train / {int(te.sum())} stream patches")
        for nm, m in [("train", ~te), ("test", te)]:
            print(f"    {nm}: acc0={(R_tr if nm == 'train' else R)[:, 0].mean():.4f} "
                  f"acc1={(R_tr if nm == 'train' else R)[:, 1].mean():.4f}  "
                  f"hospitals={dict(zip(*np.unique(centers[m], return_counts=True)))}")
        meta_extra = {"router_n_train": int((~te).sum())}

    T = len(X)
    acc0, acc1 = R[:, 0].mean(), R[:, 1].mean()
    specs = policy_specs(args.setup, T)
    print(f"  setup={args.setup}  T={T}  Agent 0 {name0} acc={acc0:.4f} | Agent 1 {name1} acc={acc1:.4f}")
    print(f"  policies: {list(specs)}")

    log_path = Path(args.log_dir) / f"camelyon17_baselines_{args.setup}_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"experiment": f"camelyon17_baselines_{args.setup}", "agent0": name0, "agent1": name1,
            "center_a": args.hospital_a, "center_b": args.hospital_b, "ood_center": args.ood_center,
            "seed": args.seed, "T": T, "eta_sqrt_A_T": float(np.sqrt(2 / T)), **meta_extra}
    _ENV.update(X=X, R=R, seed=args.seed, specs=specs)

    capacities = [float(c) for c in args.capacities.split(",")]
    tasks = [(r, c, l) for r in range(args.n_runs) for c in capacities for l in specs]
    results = {}  # (run, cap) -> (policies, tracks)
    step_actions = {l: {c: [None] * args.n_runs for c in capacities} for l in specs}

    def write(f, rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()

    t0 = time.time()
    with open(log_path, "w") as f:
        write(f, {"capacity": 0.0, "run": -1, "error_rate": 1.0 - acc1, **meta})
        write(f, {"capacity": 1.0, "run": -1, "error_rate": 1.0 - acc0, **meta})
        if args.setup == "split":
            for kind, nm in [("log", "Unconstrained Logistic"), ("tree", "Unconstrained Tree")]:
                S = _ENV["S_" + kind]
                acc = R[np.arange(T), np.where(S[:, 0] >= S[:, 1], 0, 1)].mean()
                write(f, {"capacity": -1.0, "run": -1, "error_rate": 1.0 - acc, "policy": nm, **meta})
                print(f"  {nm} (router-train half) err={1 - acc:.4f}")

        def collect(res):
            run_idx, cap, label, avg, track, acts = res
            step_actions[label][cap][run_idx] = acts
            pol, tr = results.setdefault((run_idx, cap), ({}, {}))
            pol[label], tr[label] = avg, track
            if len(pol) == len(specs):  # all policies for this (run, cap) done -> log
                write(f, {"capacity": cap, "run": run_idx, "policies": pol, "tracking": tr,
                          "ts": datetime.utcnow().isoformat(), **meta})
            n_done = sum(len(v[0]) for v in results.values())
            print(f"  [{n_done}/{len(tasks)}]", end="\r", flush=True)

        if args.workers <= 1:
            for t in tasks:
                collect(_task(*t))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for fu in as_completed([ex.submit(_task, *t) for t in tasks]):
                    collect(fu.result())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Log: {log_path}")
    if args.save_actions:
        # Step-level actions {policy: {capacity_agent0: (n_runs, T) int8}} and run-level CSV,
        # in the same layout as the Bank result files (runs numbered from 1).
        out = Path(args.save_actions)
        out.mkdir(parents=True, exist_ok=True)
        tag = f"camelyon17_seed_{args.seed}_runs{args.n_runs}_{args.setup}"
        np.save(out / f"{tag}_step_actions.npy",
                {l: {c: np.stack(v) for c, v in d.items()} for l, d in step_actions.items()})
        pd.DataFrame([{"run": r + 1, "capacity_agent0": c, "capacity_agent1": 1 - c, "policy": l,
                       "avg_reward": pol[l], "error_rate": 1 - pol[l]}
                      for (r, c), (pol, _) in results.items() for l in pol]
                     ).sort_values(["policy", "capacity_agent0", "run"]).to_csv(out / f"{tag}_raw_runs.csv", index=False)
        print(f"Saved step actions + raw runs to {out}/{tag}_*")
    return log_path


def summarize(log_path: Path) -> None:
    recs = [json.loads(l) for l in open(log_path) if l.strip()]
    rows = [{"cap_agent1": round(1 - r["capacity"], 2), "policy": k, "err": 1 - v,
             "share0": r["tracking"][k]["share0"], "max_dev": r["tracking"][k]["max_dev"],
             "target0": r["capacity"]}
            for r in recs if r["run"] >= 0 for k, v in r["policies"].items()]
    d = pd.DataFrame(rows)
    order = list(dict.fromkeys(d["policy"]))
    g = d.groupby(["policy", "cap_agent1"])["err"]
    tab = (g.mean().map("{:.4f}".format) + " ± " + (g.std() / np.sqrt(g.count())).map("{:.4f}".format)).unstack()
    n = d.groupby(["policy", "cap_agent1"]).size().min()
    print(f"\n{log_path.name}: error rate, mean ± SE over {n} runs (columns = capacity of Agent 1)")
    print(tab.loc[order].to_string())
    for r in recs:
        if r["run"] == -1:
            lbl = r.get("policy") or ("Agent 1 only" if r["capacity"] == 0.0 else "Agent 0 only")
            print(f"  {lbl:24s} err={r['error_rate']:.4f}")
    d["share_err"] = (d["share0"] - d["target0"]).abs()
    print("Capacity tracking (mean over runs): |final share0 - target| / max_t>=100 |share0_t - target|")
    t2 = d.groupby(["policy", "cap_agent1"]).apply(
        lambda x: f"{x.share_err.mean():.3f}/{x.max_dev.mean():.3f}", include_groups=False).unstack()
    print(t2.loc[order].to_string())


def plot(log_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import load_log, build_plot_data

    data = load_log(log_path)
    xy = build_plot_data(data)
    b = data["boundaries"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([1.0, 0.0], [b[0.0], b[1.0]], color="gray", linestyle="--", linewidth=2, label="Non-contextual")
    style = {"Logistic": ("#1f77b4", "o"), "Tree": ("#d62728", "s")}
    ls_cycle = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    seen = {"Logistic": 0, "Tree": 0}
    for pol in xy:
        if pol == "Random":
            continue
        fam = "Tree" if "Tree" in pol else "Logistic"
        color, marker = style[fam]
        ls = ls_cycle[seen[fam] % len(ls_cycle)]
        seen[fam] += 1
        xs, ys = xy[pol]
        o = np.argsort(xs)
        ax.plot(np.array(xs)[o], np.array(ys)[o], marker=marker, linestyle=ls, linewidth=1.8,
                color=color, alpha=0.6 + 0.4 * (seen[fam] == 1), label=pol)
    for pol, err in data["unconstrained"].items():
        ax.axhline(err, color="#f6b26b" if "Logistic" in pol else "#93d17c", linestyle="--", linewidth=1.5, label=pol)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(r"$\alpha_1$: Capacity of Agent 1", fontsize=16)
    ax.set_ylabel("Error Rate", fontsize=16)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    out = ROOT / "plots" / (log_path.stem + ".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", choices=["baselines", "eta0", "split"])
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--capacities", default="0.2,0.4,0.5,0.6,0.8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--hospital-a", type=int, default=3)
    ap.add_argument("--hospital-b", type=int, default=4)
    ap.add_argument("--ood-center", type=int, default=1)
    ap.add_argument("--sim-samples", type=int, default=2000)
    ap.add_argument("--save-actions", default=None, help="directory for step-action .npy + raw-runs .csv")
    ap.add_argument("--plot", default=None, help="summarise + plot an existing log and exit")
    args = ap.parse_args()
    if args.plot:
        summarize(Path(args.plot))
        print(f"Plot saved to {plot(Path(args.plot))}")
        return
    if args.setup is None:
        ap.error("--setup is required")
    lp = run(args)
    summarize(lp)
    print(f"Plot saved to {plot(lp)}")


if __name__ == "__main__":
    main()
