"""
Half-split benchmark: offline router vs online policies on the SAME online stream.

Online policies are identical to main_100runs_error_rate.py (eta = 0.5); the offline routers
are logistic / random-forest models of each agent's correctness, frozen at deployment.

Data protocol
-------------
  70% training split  -> fit the two expert agents only (logistic regression + XGBoost), as in the main runs.
  30% held-out split  -> split 50/50 (fixed, random_state=42):
      H_off ("router-train half"): labels used ONLY to fit the offline routers
                                    (logistic + random forest, one per agent) on each agent's correctness.
      H_on  ("online half")       : the online stream. Every policy is evaluated here.

Policies (all use the virtual queue with eta = 0.5, capacities 0.2/0.4/0.5/0.6/0.8):
  Offline Logistic / Offline Tree : routers fitted on H_off, frozen on H_on (queue update only).
  Logistic TS / Logistic greedy / Tree TS / Tree greedy :
      start with no expertise model and learn ONLY from bandit feedback on the H_on stream.

10 runs; run r uses permutation seed (seed + r) of H_on, shared by every policy and capacity,
so offline vs online comparisons are paired.

Usage (from anywhere):  python bank/code/half_split_offline_router_10runs.py [seed]   (default and paper setting: seed 2)
Data:   bank/data/bank_marketing_raw.csv
Output: bank/code/results_half_split_10_run/  (saved copy: bank/results/half_split_offline_router_10runs/)
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ============================================================
# 0. ARGS / PATHS
# ============================================================
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng = np.random.default_rng(seed)
np.random.seed(seed)

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "data" / "bank_marketing_raw.csv"   

OUT_DIR = HERE / f"results_half_split_10_run"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"\n=== Running with seed = {seed} ===")
print(f"Data: {DATA_PATH}\nOutput: {OUT_DIR}\n")

# ============================================================
# 1. FEATURE SPLIT (identical to the main script)
# ============================================================
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                        'contact', 'month', 'poutcome', 'day_of_week']
continuous_features = ['age', 'duration', 'campaign', 'pdays', 'previous']


def split_features_randomly(cat_features, cont_features, rng, frac=0.2):
    all_features = cat_features + cont_features
    n_agent1 = max(1, int(np.floor(frac * len(all_features))))
    agent1_feats = rng.choice(all_features, size=n_agent1, replace=False).tolist()
    agent2_feats = [f for f in all_features if f not in agent1_feats]
    return ([f for f in agent1_feats if f in cat_features], [f for f in agent1_feats if f in cont_features],
            [f for f in agent2_feats if f in cat_features], [f for f in agent2_feats if f in cont_features])


agent1_cat, agent1_cont, agent2_cat, agent2_cont = split_features_randomly(
    categorical_features, continuous_features, rng)
context_cat = agent1_cat + agent2_cat
context_cont = agent1_cont + agent2_cont
print("Agent 1 features:", agent1_cat, agent1_cont)
print("Agent 2 features:", agent2_cat, agent2_cont)


# ============================================================
# 2. DATA (identical encoding + 70/30 split as the main script)
# ============================================================
def block(df, cat, cont, n):
    Xc = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df[cat]) if cat else np.empty((n, 0))
    Xn = StandardScaler().fit_transform(df[cont]) if cont else np.empty((n, 0))
    return np.hstack([Xc, Xn])


df = pd.read_csv(DATA_PATH, sep=';')
df["target"] = (df["y"] == "yes").astype(int)
n = len(df)
X1 = block(df, agent1_cat, agent1_cont, n)
X2 = block(df, agent2_cat, agent2_cont, n)
Xc = np.hstack([OneHotEncoder(sparse_output=False).fit_transform(df[context_cat]),
                StandardScaler().fit_transform(df[context_cont])])

Xc_train, Xc_test, X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    Xc, X1, X2, df["target"].values, test_size=0.3, random_state=42)

# ============================================================
# 3. EXPERT AGENTS (trained on the 70% split, as before)
# ============================================================
def make_lr():
    return LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)


def make_xgb(rs):
    return xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric='logloss', random_state=rs)


# Agent models: code agent 1 = logistic regression (paper Agent 2),
#               code agent 2 = XGBoost (paper Agent 1).
model_agent1, model_agent2 = make_lr(), make_xgb(seed)

model_agent1.fit(X1_train, y_train)
model_agent2.fit(X2_train, y_train)

# Correctness of each agent on every held-out task
rewards_matrix_test = np.column_stack([
    (model_agent1.predict(X1_test) == y_test).astype(int),
    (model_agent2.predict(X2_test) == y_test).astype(int),
])

# ============================================================
# 4. SPLIT THE 30% HELD-OUT SET INTO HALVES
# ============================================================
idx_test = np.arange(len(y_test))
idx_off, idx_on = train_test_split(idx_test, test_size=0.5, random_state=42)

Xc_off, R_off = Xc_test[idx_off], rewards_matrix_test[idx_off]   # router-train half
Xc_on, R_on = Xc_test[idx_on], rewards_matrix_test[idx_on]       # online half

print(f"\nHeld-out size {len(idx_test)} -> router-train half {len(idx_off)}, online half {len(idx_on)}")
for name, R in [("router-train half", R_off), ("online half", R_on)]:
    print(f"  {name}: agent1 acc = {R[:, 0].mean():.4f}, agent2 acc = {R[:, 1].mean():.4f}")

# ============================================================
# 5. OFFLINE ROUTERS: fitted on H_off only
# ============================================================
offline_log = [LogisticRegression(solver='lbfgs', max_iter=1000).fit(Xc_off, R_off[:, k]) for k in range(2)]
offline_tree = [RandomForestRegressor(n_estimators=20, max_depth=3, min_samples_leaf=10,
                                      random_state=42).fit(Xc_off, R_off[:, k]) for k in range(2)]

# Precompute frozen router predictions on H_on (routers never change, so this is exact and fast)
offline_pred_on = {
    "offline_logistic": np.column_stack([m.predict_proba(Xc_on)[:, 1] for m in offline_log]),
    "offline_tree": np.column_stack([m.predict(Xc_on) for m in offline_tree]),
}
print("Offline routers trained on the router-train half only.")


# ============================================================
# 6. ONLINE AGENT CLASSES (copied from the main script)
# ============================================================
class ContextualLogRegAgent:
    def __init__(self, d, alpha=1.0, lambda_prior=1.0):
        self.d = d
        self.alpha = alpha
        self.m = np.zeros(d)
        self.A_inv = np.eye(d) / lambda_prior

    def sample_prob(self, x):
        cov = (self.alpha ** 2) * self.A_inv
        cov = (cov + cov.T) / 2.0
        try:
            theta = np.random.multivariate_normal(self.m, cov)
        except np.linalg.LinAlgError:
            theta = np.random.multivariate_normal(self.m, cov + 1e-6 * np.eye(self.d))
        return expit(np.dot(x, theta))

    def predict_reward_prob(self, x):
        return expit(np.dot(self.m, x))

    def update(self, x, reward):
        p_hat = expit(np.dot(self.m, x))
        w = np.clip(p_hat * (1 - p_hat), 1e-4, 0.25)
        Ax = self.A_inv @ x
        self.A_inv = self.A_inv - w * np.outer(Ax, Ax) / (1 + w * np.dot(x, Ax))
        self.A_inv = (self.A_inv + self.A_inv.T) / 2.0
        self.m = self.m + self.A_inv @ ((reward - p_hat) * x)


class AgentPosteriorRewardTreeTS:
    def __init__(self, n_trees=20, max_depth=3, min_samples_leaf=10, refit_every=20, rng=None):
        self.n_trees = n_trees
        self.refit_every = refit_every
        self.fitted = False
        self.rng = rng if rng is not None else np.random.default_rng()
        self.X, self.y = [], []
        self.trees = [DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                      for _ in range(n_trees)]
        self.t = 0

    def sample_reward(self, x):
        if not self.fitted:
            return self.rng.random()
        idx = self.rng.integers(self.n_trees)
        return float(np.clip(self.trees[idx].predict(x.reshape(1, -1))[0], 0.0, 1.0))

    def predict_reward(self, x):
        if not self.fitted:
            return 0.5
        preds = [tree.predict(x.reshape(1, -1))[0] for tree in self.trees]
        return float(np.clip(np.mean(preds), 0.0, 1.0))

    def update(self, x, r):
        self.X.append(x)
        self.y.append(r)
        self.t += 1
        if self.t % self.refit_every != 0:
            return
        X_arr, y_arr = np.array(self.X), np.array(self.y)
        for tree in self.trees:
            idx = self.rng.integers(0, len(y_arr), size=len(y_arr))
            tree.fit(X_arr[idx], y_arr[idx])
        self.fitted = True


# ============================================================
# 7. SIMULATORS
# ============================================================
def run_online(X, rewards_matrix, p_targets, perm, eta, mode):
    """One online run on the permuted stream; agents learn from bandit feedback only."""
    Xs, Rs = X[perm], rewards_matrix[perm]
    n_steps, dim = Xs.shape
    n_k = len(p_targets)
    agent_rng = np.random.default_rng(42)  # same as main script (rng created inside run_simulation)
    if mode in ('tree_ts', 'tree_greedy'):
        agents = [AgentPosteriorRewardTreeTS(rng=agent_rng) for _ in range(n_k)]
    else:
        agents = [ContextualLogRegAgent(dim, alpha=0.5) for _ in range(n_k)]

    Q = np.zeros(n_k)
    step_rewards = np.zeros(n_steps)
    step_actions = np.zeros(n_steps, dtype=int)
    for t in range(n_steps):
        x_t = Xs[t]
        if mode == 'tree_ts':
            est = np.array([a.sample_reward(x_t) for a in agents])
        elif mode == 'tree_greedy':
            est = np.array([a.predict_reward(x_t) for a in agents])
        elif mode == 'log_ts':
            est = np.array([a.sample_prob(x_t) for a in agents])
        elif mode == 'log_greedy':
            est = np.array([a.predict_reward_prob(x_t) for a in agents])
        else:
            raise ValueError(mode)
        action = int(np.argmax(est - eta * Q))
        reward = Rs[t, action]
        step_actions[t], step_rewards[t] = action, reward
        agents[action].update(x_t, reward)
        z = np.zeros(n_k); z[action] = 1.0
        Q = np.maximum(Q + z - p_targets, 0.0)
    return step_rewards.mean(), step_rewards, step_actions


def run_offline(pred, rewards_matrix, p_targets, perm, eta):
    """Frozen router on the permuted stream; only the virtual queue changes."""
    P, Rs = pred[perm], rewards_matrix[perm]
    n_steps, n_k = Rs.shape
    Q = np.zeros(n_k)
    step_rewards = np.zeros(n_steps)
    step_actions = np.zeros(n_steps, dtype=int)
    for t in range(n_steps):
        action = int(np.argmax(P[t] - eta * Q))
        step_actions[t], step_rewards[t] = action, Rs[t, action]
        z = np.zeros(n_k); z[action] = 1.0
        Q = np.maximum(Q + z - p_targets, 0.0)
    return step_rewards.mean(), step_rewards, step_actions


# Capacity tracking (same definitions as the eta-theory / DMD scripts)
CHECKPOINTS = [0.1, 0.25, 0.5, 0.75, 1.0]
BURN_IN = 100


def capacity_tracking(actions, cap0):
    n_s = len(actions)
    share0 = np.cumsum(actions == 0) / np.arange(1, n_s + 1)
    dev = np.abs(share0[BURN_IN - 1:] - cap0)
    out = {"realized_frac_agent0": float(share0[-1]),
           "max_abs_dev_after_burnin": float(dev.max()),
           "mean_abs_dev_after_burnin": float(dev.mean())}
    for c in CHECKPOINTS:
        out[f"share0_at_{int(c * 100)}pct"] = float(share0[max(1, int(np.floor(c * n_s))) - 1])
    return out


# ============================================================
# 8. EXPERIMENT LOOP: 10 RUNS
# ============================================================
capacities = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
n_runs = 10
eta_val = 0.5

policy_modes = {
    "Offline Logistic": "offline_logistic",
    "Offline Tree": "offline_tree",
    "Logistic TS": "log_ts",
    "Logistic greedy": "log_greedy",
    "Tree TS": "tree_ts",
    "Tree greedy": "tree_greedy",
}

run_rows, permutations = [], []
step_rewards_accum = {p: {float(c): [] for c in capacities} for p in policy_modes}
step_actions_accum = {p: {float(c): [] for c in capacities} for p in policy_modes}
tag = f"banking_seed_{seed}_runs{n_runs}_half_split"

for r in range(1, n_runs + 1):
    print(f"\n=== Run {r}/{n_runs} ===", flush=True)
    perm_r = np.random.default_rng(seed + r).permutation(len(idx_on))
    permutations.append(perm_r)

    for cap in capacities:
        targets = np.array([cap, 1.0 - cap])
        for policy_name, mode_name in policy_modes.items():
            if mode_name.startswith("offline"):
                avg_reward, sr, sa = run_offline(offline_pred_on[mode_name], R_on, targets, perm_r, eta_val)
            else:
                avg_reward, sr, sa = run_online(Xc_on, R_on, targets, perm_r, eta_val, mode_name)
            run_rows.append({
                "seed": seed, "run": r,
                "capacity_agent0": float(cap), "capacity_agent1": float(1.0 - cap),
                "eta": eta_val, "policy": policy_name, "mode": mode_name,
                "n_router_train": len(idx_off), "n_online": len(idx_on),
                "avg_reward": avg_reward, "error_rate": 1.0 - avg_reward,
                **capacity_tracking(sa, cap),
            })
            step_rewards_accum[policy_name][float(cap)].append(sr)
            step_actions_accum[policy_name][float(cap)].append(sa)
            print(f"  cap={cap:.1f} {policy_name:16s} err={1.0 - avg_reward:.4f}", flush=True)

    # Save raw run-level CSV after every run so partial progress is kept
    pd.DataFrame(run_rows).to_csv(OUT_DIR / f"{tag}_raw_runs.csv", index=False)

# ============================================================
# 9. SAVE EVERYTHING
# ============================================================
run_df = pd.DataFrame(run_rows)
raw_path = OUT_DIR / f"{tag}_raw_runs.csv"
run_df.to_csv(raw_path, index=False)

summary_df = (
    run_df.groupby(["capacity_agent0", "capacity_agent1", "policy"], as_index=False)
    .agg(n_runs=("error_rate", "count"),
         mean_reward=("avg_reward", "mean"),
         mean_error=("error_rate", "mean"),
         sd_error=("error_rate", "std"),
         se_error=("error_rate", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
         mean_realized_frac_agent0=("realized_frac_agent0", "mean"))
)
summary_df["ci95_halfwidth"] = 1.96 * summary_df["se_error"]
summary_path = OUT_DIR / f"{tag}_summary.csv"
summary_df.to_csv(summary_path, index=False)

stack = lambda acc: {p: {c: np.stack(v, axis=0) for c, v in d.items()} for p, d in acc.items()}
np.save(OUT_DIR / f"{tag}_step_rewards.npy", stack(step_rewards_accum))
np.save(OUT_DIR / f"{tag}_step_actions.npy", stack(step_actions_accum))
np.save(OUT_DIR / f"{tag}_permutations.npy", np.stack(permutations, axis=0))
# Indices into the 30% held-out set (order of train_test_split(..., test_size=0.3, random_state=42))
np.savez(OUT_DIR / f"{tag}_split_indices.npz", idx_router_train=idx_off, idx_online=idx_on)

results_json = {
    "meta": {"seed": seed, "eta": eta_val, "n_runs": n_runs,
             "capacities_agent0": capacities.tolist(), "n_router_train": int(len(idx_off)),
             "n_online": int(len(idx_on)),
             "agent_acc_online_half": R_on.mean(axis=0).tolist(),
             "agent_acc_router_train_half": R_off.mean(axis=0).tolist()},
    "mean_error": {p: [float(summary_df.loc[(summary_df.capacity_agent0 == float(c)) & (summary_df.policy == p),
                                             "mean_error"].iloc[0]) for c in capacities] for p in policy_modes},
}
json_path = OUT_DIR / f"{tag}_results.json"
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)

print("\nFinished. Outputs in", OUT_DIR)
for p in sorted(OUT_DIR.glob(f"{tag}_*")):
    print("  ", p.name)
