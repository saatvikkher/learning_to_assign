# Dual mirror descent (Euclidean and entropic) baseline on Bank, 10 runs (same task orders as the main runs).
# Data preparation, feature split, agents and online reward models are identical to main_100runs_error_rate.py.
# Run from this folder: python -u dmd_10runs.py   (default seed 2 = paper setting)

# Constrained-bandit baseline for Bank: Dual Mirror Descent (DMD) for online allocation
# (Balseiro, Lu & Mirrokni, "The Best of Many Worlds: Dual Mirror Descent for Online
# Allocation Problems", Oper. Res. 2023 / ICML 2020), run with bandit feedback by plugging
# in the same online reward models as our policies. 10 runs, results saved after run 5 and 10.
# Realized capacity is tracked at every step (saved actions, checkpoints, dual-variable paths).
#
# New code starts at 'DUAL MIRROR DESCENT BASELINE'.
#
# Optional args: seed (default 2), number of runs (default 10), truncate T (smoke tests only).


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.special import expit
import time
from sklearn.preprocessing import OneHotEncoder
import sys
import json
import xgboost as xgb
# ==========================================
# 1. DATA LOADING & PREP
# ==========================================

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng = np.random.default_rng(seed)
np.random.seed(seed)

print(f"\n=== Running with seed = {seed} ===\n")


# ============================================================
# 1. FEATURE DEFINITIONS
# ============================================================



categorical_features = [
    'job',        # type of job
    'marital',    # marital status
    'education',  # education level
    'default',    # credit in default? (yes/no)
    'housing',    # housing loan? (yes/no)
    'loan',       # personal loan? (yes/no)
    'contact',    # contact communication type
    'month',      # last contact month
    'poutcome' ,   # outcome of previous campaign
    'day_of_week' # mon friday...
]

continuous_features = [
    'age',        # numeric
    'duration',   # last contact duration (seconds)
    'campaign',   # number of contacts during this campaign
    'pdays',      # days since last contact (-1 if never)
    'previous'    # number of contacts before this campaign
]

def split_features_randomly(cat_features, cont_features, rng, frac=0.2):
    all_features = cat_features + cont_features
    n_total = len(all_features)
    n_agent1 = max(1, int(np.floor(frac * n_total)))

    # sample 20% from all features
    agent1_feats = rng.choice(all_features, size=n_agent1, replace=False).tolist()
    agent2_feats = [f for f in all_features if f not in agent1_feats]

    # split back by type
    agent1_cat = [f for f in agent1_feats if f in cat_features]
    agent1_cont = [f for f in agent1_feats if f in cont_features]

    agent2_cat = [f for f in agent2_feats if f in cat_features]
    agent2_cont = [f for f in agent2_feats if f in cont_features]

    return agent1_cat, agent1_cont, agent2_cat, agent2_cont


agent1_cat, agent1_cont, agent2_cat, agent2_cont = split_features_randomly(
    categorical_features, continuous_features, rng
)

print("Agent 1 features:")
print("  Categorical:", agent1_cat)
print("  Continuous :", agent1_cont)

print("\nAgent 2 features:")
print("  Categorical:", agent2_cat)
print("  Continuous :", agent2_cont)

context_cat = agent1_cat + agent2_cat
context_cont = agent1_cont + agent2_cont
print('the context Categorical: ', context_cat)
print('the context Continouous: ', context_cont)
# ============================================================
# 2. DATA LOADING & PREP
# ============================================================
def load_and_prep_data(agent1_cat, agent1_cont, agent2_cat, agent2_cont):
    df = pd.read_csv('../data/bank_marketing_raw.csv', sep=';')


    df["target"] = (df["y"] == "yes").astype(int)


    n = len(df)

    # ---------- Agent 1 ----------
    if len(agent1_cat) > 0:
        X1_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore') \
            .fit_transform(df[agent1_cat])
    else:
        X1_cat = np.empty((n, 0))
    
    if len(agent1_cont) > 0:
        X1_cont = StandardScaler().fit_transform(df[agent1_cont])
    else:
        X1_cont = np.empty((n, 0))
    
    X1 = np.hstack([X1_cat, X1_cont])
    
    
    # ---------- Agent 2 ----------
    if len(agent2_cat) > 0:
        X2_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore') \
            .fit_transform(df[agent2_cat])
    else:
        X2_cat = np.empty((n, 0))
    
    if len(agent2_cont) > 0:
        X2_cont = StandardScaler().fit_transform(df[agent2_cont])
    else:
        X2_cont = np.empty((n, 0))
    
    X2 = np.hstack([X2_cat, X2_cont])


    Xc_cat = OneHotEncoder(sparse_output=False).fit_transform(df[context_cat])
    Xc_cont = StandardScaler().fit_transform(df[context_cont])
    Xc = np.hstack([Xc_cat, Xc_cont])

    return train_test_split(
        Xc, X1, X2, df["target"].values,
        test_size=0.3, random_state=42
    )


Xc_train, Xc_test, X1_train, X1_test, X2_train, X2_test, y_train, y_test = \
    load_and_prep_data(agent1_cat, agent1_cont, agent2_cat, agent2_cont)

# ==========================================
# 2. TRAIN EXPERTS & OFFLINE POLICIES
# ==========================================
print("--- Step 2: Training Expert Arms & Offline Benchmarks ---")

# --------------------------------------------------
# EXPERT MODELS
# --------------------------------------------------

# Agent models: code agent 1 = logistic regression (paper Agent 2),
#               code agent 2 = XGBoost (paper Agent 1).
model_agent1 = LogisticRegression(
    solver='lbfgs',
    class_weight='balanced',
    max_iter=1000
)
model_agent2 = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=seed
)


model_agent1.fit(X1_train, y_train)
model_agent2.fit(X2_train, y_train)


# ============================================================
# TEST ENVIRONMENT
# ============================================================
preds_1_test = model_agent1.predict(X1_test)
preds_2_test = model_agent2.predict(X2_test)

rewards_matrix_test = np.column_stack([
    (preds_1_test == y_test).astype(int),
    (preds_2_test == y_test).astype(int)
])

print('agent 1 test marginal accuracy:', rewards_matrix_test[:, 0].mean())
print('agent 2 test marginal accuracy:', rewards_matrix_test[:, 1].mean())



# ==========================================
# 4. AGENT CLASSES 
# ==========================================
class ContextualLogRegAgent:
        def __init__(self, d, alpha=1.0, lambda_prior=1.0):
            self.d = d
            self.alpha = alpha 
            self.m = np.zeros(d)
            # Initialize with ridge regularization
            self.A_inv = np.eye(d) / lambda_prior

        def sample_prob(self, x):
            # Calculate covariance
            cov = (self.alpha**2) * self.A_inv
            
            # 1. Enforce Symmetry (Fixes small numerical drift)
            cov = (cov + cov.T) / 2.0
            
            # 2. Robust Sampling with Fallback
            try:
                theta = np.random.multivariate_normal(self.m, cov)
            except np.linalg.LinAlgError:
                # If SVD/Cholesky fails, matrix is probably not positive definite due to float errors.
                # this is the small epsilon you can add 
                jitter = 1e-6 * np.eye(self.d)
                theta = np.random.multivariate_normal(self.m, cov + jitter)

            return expit(np.dot(x, theta))

        def predict_reward_prob(self, x):

            return expit(np.dot(self.m, x))
        
        def get_ucb_prob(self, x, alpha_ucb=0.5):
            # UCB doesn't need sampling, just variance calculation
            # We clamp variance to be non-negative just in case
            var = np.dot(x, self.A_inv @ x)
            uncertainty = np.sqrt(max(var, 1e-10)) 
            return expit(np.dot(self.m, x) + alpha_ucb * uncertainty)

        def update(self, x, reward):
            # Laplace Approximation / Sherman-Morrison Update
            p_hat = expit(np.dot(self.m, x))
            
            # Weight w
            w = p_hat * (1 - p_hat)
            # Safety clip to prevent division by zero or extreme updates
            w = np.clip(w, 1e-4, 0.25) 
            
            Ax = self.A_inv @ x
            numerator = w * np.outer(Ax, Ax)
            denominator = 1 + w * np.dot(x, Ax)
            
            self.A_inv = self.A_inv - numerator / denominator
            
            # this should help fix bugs lmk if it doesn't work
            self.A_inv = (self.A_inv + self.A_inv.T) / 2.0
            
            g = (reward - p_hat) * x
            self.m = self.m + self.A_inv @ g

class AgentPosteriorRewardTreeTS:
    """
    Bootstrapped tree ensemble for Thompson Sampling.
    """
    def __init__(self, n_trees=20, max_depth=3, min_samples_leaf=10, refit_every=20, rng=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.refit_every = refit_every
        self.fitted = False
        self.rng = rng if rng is not None else np.random.default_rng()

        self.X = []
        self.y = []
        
        self.trees = [
            DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            for _ in range(n_trees)
        ]
        self.t = 0

    def sample_reward(self, x):
        if not self.fitted:
            return self.rng.random()
        idx = self.rng.integers(self.n_trees)
        return float(np.clip(self.trees[idx].predict(x.reshape(1, -1))[0], 0.0, 1.0))


    def predict_reward(self, x):
        if not self.fitted:
            return 0.5  # prior mean
    
        preds = [self.trees[j].predict(x.reshape(1, -1))[0] for j in range(self.n_trees)]
        return float(np.clip(np.mean(preds), 0.0, 1.0))
        
    def update(self, x, r):
        self.X.append(x) 
        self.y.append(r)
        self.t += 1

        if self.t % self.refit_every != 0:
            return

        X_arr = np.array(self.X)
        y_arr = np.array(self.y)
        n = len(y_arr)
        
        for tree in self.trees:
            idx = self.rng.integers(0, n, size=n)
            tree.fit(X_arr[idx], y_arr[idx])
        self.fitted = True




# ============================================================
# DUAL MIRROR DESCENT BASELINE (new code)
# ============================================================
# Online allocation with resource constraints (Balseiro, Lu & Mirrokni 2023, Algorithm 1).
# Resource k = agent k's capacity; budget B_k = rho_k * T, where rho = (cap0, 1 - cap0) is the
# same p_targets vector our queue policies use (index 0 = LR agent, index 1 = XGBoost agent,
# i.e. paper Agent 2 / Agent 1 respectively; see README.md).
#
# At each step t:
#   1. score_k = mu_hat_k(x_t) - lambda_k        (mu_hat = greedy point estimate of agent k's accuracy)
#   2. pick a_t = argmax over agents whose budget is not exhausted
#      (hard budgets: once agent k has received B_k tasks it is unavailable; since every task must
#       be routed and rho_0 + rho_1 = 1, the other agent then takes all remaining tasks)
#   3. observe r_t only for a_t (bandit feedback), update that agent's reward model
#   4. dual subgradient g_t = rho - e_{a_t};  mirror-descent step on lambda:
#        Euclidean: lambda <- clip(lambda - eta * g_t, 0, lambda_max)
#        Entropic : lambda <- clip(lambda * exp(-eta * g_t), lambda_min, lambda_max)  (multiplicative weights)
#      eta = 1/sqrt(T) (the paper's step size), lambda_max = rewardmax / min(rho) = 1 / min(rho).
#
# Relation to our method: our queue rule Q <- max(Q + e_a - rho, 0) with score mu - eta*Q is the
# Euclidean variant with lambda = eta*Q, step eta = 0.5, and no hard budget. So the Euclidean
# variant here differs from ours only in step size (1/sqrt(T)) and the hard budget stop; the
# entropic variant is a different mirror map.
#
# Reward models: the same online models as our greedy policies (logistic Laplace model /
# bootstrapped trees, point estimates), so differences come from the allocation rule only.
#
# Realized-capacity tracking (all saved):
#   - full step-level actions (cumulative realized share of agent 0 at any t = cumsum / t)
#   - realized share of agent 0 at t/T in CHECKPOINTS, final realized share, max/mean |share_t - rho_0|
#     for t >= BURN_IN, step at which each agent's budget ran out, number of forced steps
#   - dual-variable paths lambda_t (every DUAL_EVERY steps)

import os

DMD_EPS_INIT = {"euclidean": 0.0, "entropic": 0.1}   # initial lambda (entropic needs lambda > 0)
LAMBDA_MIN = 1e-6
CHECKPOINTS = [0.1, 0.25, 0.5, 0.75, 1.0]
BURN_IN = 100
DUAL_EVERY = 50


def run_dmd(cap0, perm, kind, mirror, n_steps, eta_dual=None, hard_budget=True):
    Xs = Xc_test[perm][:n_steps]
    Rs = rewards_matrix_test[perm][:n_steps]
    n, d = Xs.shape
    rho = np.array([cap0, 1.0 - cap0])
    budget = rho * n
    eta = 1.0 / np.sqrt(n) if eta_dual is None else eta_dual
    lam_max = 1.0 / rho.min()

    rng_model = np.random.default_rng(42)                 # as in run_simulation
    if kind == 'tree':
        agents = [AgentPosteriorRewardTreeTS(rng=rng_model) for _ in range(2)]
        predict = lambda a, x: a.predict_reward(x)
    else:
        agents = [ContextualLogRegAgent(d, alpha=0.5) for _ in range(2)]
        predict = lambda a, x: a.predict_reward_prob(x)

    lam = np.full(2, DMD_EPS_INIT[mirror])
    used = np.zeros(2)
    exhausted_at = [-1, -1]
    n_forced = 0
    step_rewards = np.zeros(n)
    actions = np.zeros(n, dtype=int)
    dual_path = []

    for t in range(n):
        x = Xs[t]
        scores = np.array([predict(agents[k], x) - lam[k] for k in range(2)])
        avail = (used < budget - 1e-9) if hard_budget else np.ones(2, dtype=bool)
        if not avail.any():                               # numerical edge case at t = n-1
            avail[:] = True
        if not avail.all():
            n_forced += 1
            scores[~avail] = -np.inf
        a = int(np.argmax(scores))                        # ties -> agent 0, as np.argmax in run_simulation
        r = Rs[t, a]
        agents[a].update(x, r)
        actions[t] = a
        step_rewards[t] = r
        used[a] += 1
        for k in range(2):
            if exhausted_at[k] < 0 and used[k] >= budget[k] - 1e-9:
                exhausted_at[k] = t + 1

        g = rho.copy(); g[a] -= 1.0                       # dual subgradient rho - e_a
        if mirror == 'euclidean':
            lam = np.clip(lam - eta * g, 0.0, lam_max)
        else:
            lam = np.clip(lam * np.exp(-eta * g), LAMBDA_MIN, lam_max)
        if (t + 1) % DUAL_EVERY == 0 or t == n - 1:
            dual_path.append([t + 1, lam[0], lam[1]])

    share0 = np.cumsum(actions == 0) / np.arange(1, n + 1)
    dev = np.abs(share0[BURN_IN - 1:] - cap0) if n > BURN_IN else np.abs(share0 - cap0)
    track = {
        "realized_frac_agent0": float(share0[-1]),
        "max_abs_dev_after_burnin": float(dev.max()),
        "mean_abs_dev_after_burnin": float(dev.mean()),
        "budget_exhausted_step_agent0": exhausted_at[0],
        "budget_exhausted_step_agent1": exhausted_at[1],
        "n_forced_steps": n_forced,
        "final_lambda0": float(lam[0]), "final_lambda1": float(lam[1]),
        "eta_dual": float(eta),
    }
    for c in CHECKPOINTS:
        track[f"share0_at_{int(c * 100)}pct"] = float(share0[max(1, int(round(c * n))) - 1])
    return step_rewards.mean(), step_rewards, actions, track, np.array(dual_path)


# ============================================================
# EXPERIMENT LOOP: 10 RUNS, CHECKPOINT SAVES AFTER RUN 5 AND RUN 10
# ============================================================
capacities = np.array([0.2, 0.4, 0.5, 0.6, 0.8])           # capacity of agent index 0 (LR)
n_runs_per_point = int(sys.argv[2]) if len(sys.argv) > 2 else 10
T_full = len(Xc_test)
n_steps = min(T_full, int(sys.argv[3])) if len(sys.argv) > 3 else T_full
SAVE_AT = {5, n_runs_per_point}                          # write results after these runs

OUT_DIR = f"dmd_{n_runs_per_point}_run" + ("" if n_steps == T_full else f"_T{n_steps}")
os.makedirs(OUT_DIR, exist_ok=True)

POLICIES = [   # (name, reward model, mirror map)
    ("DMD Entropic Logistic",  "logistic", "entropic"),
    ("DMD Entropic Tree",      "tree",     "entropic"),
    ("DMD Euclidean Logistic", "logistic", "euclidean"),
    ("DMD Euclidean Tree",     "tree",     "euclidean"),
]

run_rows = []
step_rewards_accum = {p[0]: {float(c): [] for c in capacities} for p in POLICIES}
step_actions_accum = {p[0]: {float(c): [] for c in capacities} for p in POLICIES}
dual_path_accum = {p[0]: {float(c): [] for c in capacities} for p in POLICIES}


def save_results(n_done):
    """Write all results for runs 1..n_done (file names carry runs{n_done})."""
    tag = f"banking_seed_{seed}_runs{n_done}_dmd"
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(f"{OUT_DIR}/{tag}_raw_runs.csv", index=False)

    agg = dict(n_runs=("error_rate", "count"),
               mean_error=("error_rate", "mean"),
               sd_error=("error_rate", "std"),
               se_error=("error_rate", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
               mean_realized_frac_agent0=("realized_frac_agent0", "mean"),
               mean_max_abs_dev=("max_abs_dev_after_burnin", "mean"),
               mean_mean_abs_dev=("mean_abs_dev_after_burnin", "mean"),
               mean_forced_steps=("n_forced_steps", "mean"))
    for c in CHECKPOINTS:
        agg[f"mean_share0_at_{int(c * 100)}pct"] = (f"share0_at_{int(c * 100)}pct", "mean")
    summary_df = run_df.groupby(["capacity_agent0", "capacity_agent1", "policy"], as_index=False).agg(**agg)
    summary_df["ci95_halfwidth"] = 1.96 * summary_df["se_error"]
    summary_df.to_csv(f"{OUT_DIR}/{tag}_summary.csv", index=False)

    results_json = {
        name: [float(summary_df.loc[(summary_df.capacity_agent0 == float(c)) &
                                    (summary_df.policy == name), "mean_error"].iloc[0])
               for c in capacities]
        for name, _, _ in POLICIES
    }
    with open(f"{OUT_DIR}/{tag}_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # {policy: {capacity_agent0: array(n_done, T)}}; 0 = LR agent, 1 = XGBoost (same format as main results)
    np.save(f"{OUT_DIR}/{tag}_step_rewards.npy",
            {k: {c: np.stack(v) for c, v in d.items()} for k, d in step_rewards_accum.items()})
    np.save(f"{OUT_DIR}/{tag}_step_actions.npy",
            {k: {c: np.stack(v) for c, v in d.items()} for k, d in step_actions_accum.items()})
    # {policy: {capacity_agent0: array(n_done, n_points, 3)}} with columns (t, lambda0, lambda1)
    np.save(f"{OUT_DIR}/{tag}_dual_paths.npy",
            {k: {c: np.stack(v) for c, v in d.items()} for k, d in dual_path_accum.items()})

    print(f"\n--- Saved results for runs 1-{n_done} to {OUT_DIR}/{tag}_* ---")
    cols = ["capacity_agent0", "policy", "n_runs", "mean_error", "se_error",
            "mean_realized_frac_agent0", "mean_max_abs_dev", "mean_forced_steps"]
    print(summary_df[cols].round(4).to_string(index=False), flush=True)


print(f"Running dual mirror descent baselines (T = {n_steps})...")
for r in range(1, n_runs_per_point + 1):
    print(f"\n=== DMD run {r}/{n_runs_per_point} ===", flush=True)

    # Same permutation construction as the online, offline-router and plug-in scripts.
    rng_run = np.random.default_rng(seed + r)
    perm_r = rng_run.permutation(len(Xc_test))

    for cap in capacities:
        for name, kind, mirror in POLICIES:
            avg, steps, acts, track, dual_path = run_dmd(cap, perm_r, kind, mirror, n_steps)
            run_rows.append({
                "seed": seed, "run": r, "T": n_steps,
                "capacity_agent0": float(cap), "capacity_agent1": float(1.0 - cap),
                "policy": name, "reward_model": kind, "mirror": mirror,
                "avg_reward": avg, "error_rate": 1.0 - avg, **track,
            })
            step_rewards_accum[name][float(cap)].append(steps)
            step_actions_accum[name][float(cap)].append(acts)
            dual_path_accum[name][float(cap)].append(dual_path)
            print(f"  cap0={cap:.1f} {name:24s} error={1.0 - avg:.4f} "
                  f"realized frac agent0={track['realized_frac_agent0']:.4f} "
                  f"max|dev|={track['max_abs_dev_after_burnin']:.4f} "
                  f"forced={track['n_forced_steps']}", flush=True)

    if r in SAVE_AT:
        save_results(r)

print("\nFinished.")
