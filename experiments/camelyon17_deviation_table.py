"""
Tables 10 and 11 (Camelyon17): error rate (SE) and max window deviation per capacity.

Reads the step-level actions and run-level CSV written by
    python -m experiments.camelyon17_baselines --setup baselines --n-runs 10 --save-actions results/camelyon17
and reports, per method / reward model / capacity:
  - error rate in % with SE over the 10 runs (ddof=1)
  - max window deviation: per run, the max over sliding windows of WINDOW consecutive tasks
    (excluding windows that end within the first BURN_IN tasks) of
    |share of Agent 1 in the window - alpha_1|; then the max over runs.
    Same definition as window_deviation() for Tables 8 and 9 (tabular notebook, Section 11),
    which uses BURN_IN = 1000 on the longer Bank stream.

Writes results/camelyon17/table{10,11}_camelyon17_{logistic,tree}_greedy_baselines.csv in the same
layout as the tabular notebook's Tables 8 and 9 (`*` = lowest value per column, bold in the paper).

Usage:
    python -m experiments.camelyon17_deviation_table
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
CAPS = [0.2, 0.4, 0.5, 0.6, 0.8]  # capacity of action 0 (Agent 2, hospital 3); alpha_1 = 1 - c
METHODS = {  # paper label: policy-name pattern in the baselines run
    "Ours, η=√(A/T)": "{m} Greedy (eta=sqrt(A/T))",
    "Online plug-in": "Plug-in {m}",
    "DMD, Euclidean": "DMD Euclidean {m}",
    "DMD, entropic": "DMD Entropic {m}",
}
WINDOW, BURN_IN = 200, 500


def window_deviation(a, c, window=WINDOW, burn_in=BURN_IN):
    # Per run: max over sliding windows of |share of action 0 in the window - c|, over windows ending
    # after task `burn_in`. With two agents this equals Agent 1's deviation from alpha_1 = 1 - c.
    cs = np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a == 0, axis=1)], axis=1)
    share = (cs[:, window:] - cs[:, :-window]) / window  # column j = window ending at task window + j
    end = np.arange(window, a.shape[1] + 1)
    return np.abs(share[:, end > burn_in] - c).max(axis=1)


def collect(acts, raw, n_runs, window, burn_in):
    rows = []
    for lab, pat in METHODS.items():
        for m in ["Logistic", "Tree"]:
            pol = pat.format(m=m)
            for c in CAPS:
                a = acts[pol][c][:n_runs]
                e = raw[(raw.policy == pol) & (raw.capacity_agent0.round(2) == c) & (raw.run <= n_runs)].error_rate
                assert len(e) == n_runs == a.shape[0]
                rows.append({"method": lab, "model": m, "alpha": c, "err": 100 * e.mean(),
                             "se": 100 * e.std(ddof=1) / np.sqrt(n_runs),
                             "max_dev": window_deviation(a, c, window, burn_in).max()})
    return pd.DataFrame(rows)


def paper_table(df, model, window):
    x = df[df.model == model]
    methods, order = list(METHODS), CAPS[::-1]
    cols = [f"α₁={1 - c:.1f}" for c in order]
    piv = lambda v: x.pivot(index="method", columns="alpha", values=v).loc[methods][order]
    err, se, dv = piv("err"), piv("se"), piv("max_dev")
    err.columns = se.columns = dv.columns = cols
    txt = err.map("{:.2f}".format) + " (" + se.map("{:.2f}".format) + ")"
    txt = txt.where(err.ne(err.min()), txt + "*")
    dtxt = dv.map("{:.3f}".format)
    dtxt = dtxt.where(dv.ne(dv.min()), dtxt + "*")
    return pd.concat({"Error rate, % (SE)": txt, f"Max deviation over {window}-task windows": dtxt})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="results/camelyon17/camelyon17_seed_42_runs10_baselines")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--burn-in", type=int, default=BURN_IN)
    ap.add_argument("--n-runs", type=int, default=10)
    args = ap.parse_args()

    pre = ROOT / args.prefix
    acts = np.load(f"{pre}_step_actions.npy", allow_pickle=True).item()
    raw = pd.read_csv(f"{pre}_raw_runs.csv")
    df = collect(acts, raw, args.n_runs, args.window, args.burn_in)

    for num, m in [(10, "Logistic"), (11, "Tree")]:
        t = paper_table(df, m, args.window)
        out = pre.parent / f"table{num}_camelyon17_{m.lower()}_greedy_baselines.csv"
        t.to_csv(out)
        print(f"\nTable {num}: Camelyon17, {m.lower()} greedy reward model "
              f"[* = lowest error / deviation per column]  -> {out.relative_to(ROOT)}")
        print(t.to_string())


if __name__ == "__main__":
    main()
