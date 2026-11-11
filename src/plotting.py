"""
Plotting utilities. Reads a JSONL log file and generates an Error vs Capacity
plot using whatever runs have been completed so far.

Log line schema:
    {"capacity": float, "run": int, "policies": {policy_name: avg_reward}, ...}

Boundary conditions (capacity 0 and 1) are stored with run=-1 and a single
"error_rate" value instead of "policies".
"""

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # headless-safe default; callers may override before import

POLICY_COLOR = {
    "Unconstrained Logistic": "#f6b26b",
    "Unconstrained Tree": "#93d17c",
    "Offline Logistic": "#ff7f0e",
    "Offline Tree": "#2ca02c",
    "Logistic TS": "#1f77b4",
    "Logistic Greedy": "#aec7e8",
    "Tree TS": "#d62728",
    "Tree Greedy": "#ff9896",
    "Random": "gray",
}

OFFLINE_UNC_COLOR = {
    "Logistic": "#f6b26b",
    "Tree":     "#93d17c",
}

PLOT_POLICIES = [
    "Unconstrained Logistic",
    "Unconstrained Tree",
    "Logistic TS",
    "Logistic Greedy",
    "Tree TS",
    "Tree Greedy",
    "Random",
]


def load_log(log_path: str | Path) -> dict:
    """Parse a JSONL log file. Returns dict with keys 'boundaries', 'unconstrained', and 'runs'."""
    boundaries = {}
    unconstrained: dict[str, float] = {}
    runs: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("run") == -1:
                cap = rec["capacity"]
                if cap == -1.0:
                    # Unconstrained offline policy stored with capacity=-1
                    policy = rec.get("policy")
                    if policy:
                        unconstrained[policy] = rec["error_rate"]
                else:
                    boundaries[cap] = rec["error_rate"]
            else:
                cap = rec["capacity"]
                for policy, avg_reward in rec["policies"].items():
                    runs[cap][policy].append(1.0 - avg_reward)

    return {"boundaries": boundaries, "unconstrained": unconstrained, "runs": runs}


def build_plot_data(log_data: dict) -> dict[str, tuple[list, list]]:
    """
    Aggregate run data per policy. Returns {policy: (x_values, y_values)}.
    x is capacity of the stronger agent (Agent 1).
    """
    boundaries = log_data["boundaries"]
    runs = log_data["runs"]

    # Sorted simulated capacities (cap = weaker agent fraction)
    # x-axis is fraction for the stronger agent = 1 - cap_weaker
    sim_caps = sorted(runs.keys())

    policy_xy: dict[str, tuple[list, list]] = {}

    all_policies = set()
    for cap_data in runs.values():
        all_policies.update(cap_data.keys())

    for policy in all_policies:
        xs = []
        ys = []
        # Left boundary: stronger gets 100% → x=1.0
        if 0.0 in boundaries:
            xs.append(1.0)
            ys.append(boundaries[0.0])
        for cap in sim_caps:
            if policy in runs[cap] and runs[cap][policy]:
                xs.append(1.0 - cap)
                ys.append(float(np.mean(runs[cap][policy])))
        # Right boundary: stronger gets 0% → x=0.0
        if 1.0 in boundaries:
            xs.append(0.0)
            ys.append(boundaries[1.0])
        if xs:
            policy_xy[policy] = (xs, ys)

    return policy_xy


def make_plot(
    log_path: str | Path,
    out_path: str | Path | None = None,
    show: bool = False,
    xlabel: str = "Capacity (Hospital 1 Agent)",
) -> Path:
    """
    Read log_path and render Error vs Capacity plot.
    Saves to out_path (default: same stem as log with .pdf in plots/).
    Returns the saved path.
    """
    log_path = Path(log_path)
    log_data = load_log(log_path)
    policy_xy = build_plot_data(log_data)

    if not policy_xy:
        raise ValueError(f"No data found in {log_path}")

    if out_path is None:
        out_path = log_path.parent.parent / "plots" / (log_path.stem + ".pdf")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=((6, 4.5)))

    # Unconstrained policies as horizontal lines — different dash styles to distinguish overlaps
    unconstrained_styles = {
        "Unconstrained Logistic": "--",
        "Unconstrained Tree": ":",
    }
    for policy in ["Unconstrained Logistic", "Unconstrained Tree"]:
        err = log_data["unconstrained"].get(policy)
        if err is not None:
            ax.axhline(
                y=err,
                color=POLICY_COLOR.get(policy, "black"),
                linestyle=unconstrained_styles[policy],
                linewidth=2.0,
                label=policy,
            )

    # Random: straight line between the two marginal error rates (never empirical)
    boundaries = log_data["boundaries"]
    if 0.0 in boundaries and 1.0 in boundaries:
        # x=1.0 → stronger gets 100% (cap_weaker=0), x=0.0 → stronger gets 0% (cap_weaker=1)
        ax.plot(
            [1.0, 0.0],
            [boundaries[0.0], boundaries[1.0]],
            color=POLICY_COLOR["Random"],
            linestyle="--",
            linewidth=2,
            label="Non-contextual",
        )

    # Empirical policies (excluding Random and Unconstrained)
    for policy in PLOT_POLICIES:
        if policy in ("Unconstrained Logistic", "Unconstrained Tree", "Random"):
            continue
        if policy not in policy_xy:
            continue
        xs, ys = policy_xy[policy]
        if not xs:
            continue
        order = np.argsort(xs)[::-1]  # decreasing x
        xs_s = [xs[i] for i in order]
        ys_s = [ys[i] for i in order]
        ax.plot(
            xs_s,
            ys_s,
            marker="o",
            linewidth=2,
            color=POLICY_COLOR.get(policy, "black"),
            label=policy,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Error Rate", fontsize=12)
    ax.legend(loc="lower center", bbox_to_anchor=(0.67, 0.165), fontsize=11, frameon=False, ncol=2)
    
    ax.grid(alpha=0)
    fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


_DEFAULT_ONLINE_POLICIES = ["Logistic TS", "Logistic Greedy", "Tree TS", "Tree Greedy"]


def make_styled_results_plot(
    log_data: dict,
    offline_unc_errors: dict,
    assignment_ratios: list,
    out_path: str | Path,
    legend: bool = False,
    xlabel: str = r"$\alpha_1$: Capacity of Agent 1",
    figsize: tuple = (6, 4.5),
    online_policies: list | None = None,
) -> Path:
    """
    Render the styled results plot used in the paper.

    offline_unc_errors : {"Logistic": float, "Tree": float}
    assignment_ratios  : unused, kept for API compatibility
    """
    policy_xy  = build_plot_data(log_data)
    boundaries = log_data["boundaries"]

    fig, ax = plt.subplots(figsize=figsize)

    # Offline unconstrained — horizontal lines
    if "Logistic" in offline_unc_errors:
        ax.axhline(offline_unc_errors["Logistic"],
                   color=OFFLINE_UNC_COLOR["Logistic"], linestyle="--",
                   linewidth=2, label="Unconstrained Logistic")
    if "Tree" in offline_unc_errors:
        ax.axhline(offline_unc_errors["Tree"],
                   color=OFFLINE_UNC_COLOR["Tree"], linestyle=(0, (1, 2)),
                   linewidth=2, label="Unconstrained Tree")

    # Random baseline
    if 0.0 in boundaries and 1.0 in boundaries:
        ax.plot([1.0, 0.0], [boundaries[0.0], boundaries[1.0]],
                color=POLICY_COLOR["Random"], linestyle="--",
                linewidth=2, label="Non-contextual")

    # Online policies
    for pol in (online_policies if online_policies is not None else _DEFAULT_ONLINE_POLICIES):
        if pol not in policy_xy:
            continue
        xs, ys = policy_xy[pol]
        order = np.argsort(xs)[::-1]
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                marker="o", linewidth=2,
                color=POLICY_COLOR.get(pol, "black"), label=pol)

    ax.set_xlim(-0.05, 1.05)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel(xlabel, fontsize=18, fontweight="medium")
    ax.set_ylabel("Error Rate", fontsize=18, fontweight="medium")
    ax.grid(alpha=0)

    if legend:
        ax.legend(
            fontsize=11,
            loc="lower center",
            bbox_to_anchor=(0.68, 0.02),
            frameon=False,
            ncol=2,
        )

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
