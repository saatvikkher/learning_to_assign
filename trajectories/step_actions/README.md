# Per-step assignment trajectories — capacity split (0.2, 0.8)

Copies of the `*_step_actions.npy` produced by
`python -m experiments.step_trajectories --dataset <name> --n-runs 100 --capacities 0.2`,
renamed by dataset. The originals, together with the matching
`_step_rewards.npy` / `_perms.npy` (and `_demographic.npy` for civilcomments),
stay in `trajectories/` under their timestamped stems:

| file here | source stem in `trajectories/` | shape |
|---|---|---|
| `mmlu_step_actions.npy`          | `mmlu_20260916_193113`          | (1, 7, 100, 217)  |
| `camelyon17_step_actions.npy`    | `camelyon17_20260916_193105`    | (1, 7, 100, 2000) |
| `imagenet16h_step_actions.npy`   | `imagenet16h_20260916_193104`   | (1, 7, 100, 2000) |
| `civilcomments_step_actions.npy` | `civilcomments_20260921_134638` | (1, 7, 100, 5880) |

Axes are `(capacity, policy, run, step)`. Axis 0 has length 1 — only the
(0.2, 0.8) split was run: Agent 0 (weaker) targets 0.2, Agent 1 (stronger) 0.8.

Values are the assigned agent: `0` = Agent 0 (weaker), `1` = Agent 1 (stronger).
Each dataset's `*_meta.json` names the two agents and carries the same axis
labels, capacities and accuracies as the original manifest.

Policy axis order (also in `meta["policies"]` — read it from there rather than
hardcoding):

| idx | policy | idx | policy |
|---|---|---|---|
| 0 | Offline Logistic | 4 | Tree TS |
| 1 | Offline Tree     | 5 | Tree Greedy |
| 2 | Logistic TS      | 6 | Random |
| 3 | Logistic Greedy  |   | |

Steps are in arrival order, which differs per run. `perms[run]` (in
`trajectories/<stem>_perms.npy`) maps step -> row of X, so per-item analysis
needs the permutation:

```python
import json, numpy as np
A = np.load("trajectories/step_actions/civilcomments_step_actions.npy")
m = json.load(open("trajectories/step_actions/civilcomments_meta.json"))
P = np.load("trajectories/civilcomments_20260921_134638_perms.npy")
D = np.load("trajectories/civilcomments_20260921_134638_demographic.npy")

tree_ts = A[0, m["policies"].index("Tree TS")]   # (100 runs, 5880 steps)
share_black = tree_ts[D[P] == "black"].mean()    # D[P] aligns demographics to steps
```
