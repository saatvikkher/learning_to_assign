"""
Summarize a JSONL experiment log: per-capacity error rate with standard error,
error reduction relative to the non-contextual (random) baseline, and the
maximum standard error across all reported cells.

The non-contextual reference is the straight line between the two marginal
error rates (the line that is plotted), i.e.
    err_rand(alpha_1) = alpha_1 * err_agent1 + (1 - alpha_1) * err_agent2
The empirical Random policy from the log is reported alongside it.

Usage:
    python summarize_log.py logs/<file>.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

POLICIES = ["Logistic TS", "Logistic Greedy", "Tree TS", "Tree Greedy", "Random"]
CONTEXTUAL = ["Logistic TS", "Logistic Greedy", "Tree TS", "Tree Greedy"]


def summarize(log_path: Path) -> None:
    runs: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    boundaries: dict[float, float] = {}
    unconstrained: dict[str, float] = {}
    meta = {}

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run") == -1:
                if rec["capacity"] == -1.0:
                    unconstrained[rec["policy"]] = rec["error_rate"]
                else:
                    boundaries[rec["capacity"]] = rec["error_rate"]
                continue
            meta = meta or rec
            for policy, avg_reward in rec["policies"].items():
                runs[rec["capacity"]][policy].append(1.0 - avg_reward)

    # Marginal error rates: capacity 0 -> all tasks to Agent 1 (stronger)
    err1 = boundaries.get(0.0)   # Agent 1 (stronger) marginal error
    err2 = boundaries.get(1.0)   # Agent 2 (weaker)   marginal error

    print(f"Log: {log_path}")
    n_runs = max((len(v) for cap in runs.values() for v in cap.values()), default=0)
    print(f"Runs per capacity: {n_runs}   capacities: {sorted(1 - c for c in runs)}")
    if meta:
        print(f"eta={meta.get('eta')}  seed={meta.get('seed')}")
    print()
    print(f"Marginal accuracy  Agent 1 (stronger, {meta.get('agent1','?')}): "
          f"{1 - err1:.4f}  (error {err1:.4f})")
    print(f"Marginal accuracy  Agent 2 (weaker,   {meta.get('agent2', meta.get('agent0','?'))}): "
          f"{1 - err2:.4f}  (error {err2:.4f})")
    for k, v in unconstrained.items():
        print(f"{k}: error {v:.4f}")
    print()

    header = (f"{'alpha_1':>8} {'policy':<16} {'error':>8} {'SE':>8} "
              f"{'rand':>8} {'abs.red':>8} {'rel.red':>8}")
    print(header)
    print("-" * len(header))

    all_se = []
    best = {}
    for cap in sorted(runs, reverse=True):
        a1 = 1.0 - cap
        err_rand = a1 * err1 + (1.0 - a1) * err2
        for pol in POLICIES:
            vals = runs[cap].get(pol)
            if not vals:
                continue
            a = np.asarray(vals, dtype=float)
            mean, se = a.mean(), a.std(ddof=1) / np.sqrt(a.size)
            all_se.append((se, pol, a1))
            abs_red = err_rand - mean
            rel_red = abs_red / err_rand
            print(f"{a1:8.2f} {pol:<16} {mean:8.4f} {se:8.4f} {err_rand:8.4f} "
                  f"{abs_red:8.4f} {rel_red:7.2%}")
            if pol in CONTEXTUAL and (a1 not in best or mean < best[a1][1]):
                best[a1] = (pol, mean, abs_red, rel_red)
        print()

    print("Best contextual policy per capacity (vs non-contextual line):")
    for a1 in sorted(best, reverse=True):
        pol, mean, abs_red, rel_red = best[a1]
        print(f"  alpha_1={a1:.2f}  {pol:<16} error={mean:.4f}  "
              f"reduction={abs_red:.4f} ({rel_red:.2%})")
    if best:
        rels = [v[3] for v in best.values()]
        abss = [v[2] for v in best.values()]
        print(f"  mean over capacities: {np.mean(abss):.4f} abs / {np.mean(rels):.2%} rel")
        print(f"  max  over capacities: {np.max(abss):.4f} abs / {np.max(rels):.2%} rel")

    se, pol, a1 = max(all_se)
    print(f"\nMax standard error across all cells: {se:.4f} "
          f"({pol}, alpha_1={a1:.2f}, n={n_runs})")
    se_c = max(s for s in all_se if s[1] in CONTEXTUAL)
    print(f"Max standard error, contextual policies only: {se_c[0]:.4f} "
          f"({se_c[1]}, alpha_1={se_c[2]:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file")
    args = parser.parse_args()
    summarize(Path(args.log_file))


if __name__ == "__main__":
    main()
