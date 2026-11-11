# Online plug-in quantile-threshold baseline on Bank, 10 runs (same task orders as the main runs).
# Data preparation, feature split, agents and online reward models are identical to main_100runs_error_rate.py.
# Run from this folder: python -u plugin_10runs.py   (default seed 2 = paper setting)

# Plug-in quantile-threshold baselines (offline and online) for Bank, 10 runs (results saved after run 5 and run 10).
# New code starts at 'PLUG-IN QUANTILE-THRESHOLD POLICIES'.

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
# OFFLINE ROUTER TRAINING: TRAIN SET ONLY
# ============================================================
# The expert models are already fitted above. Build router labels only
# from the training split; the test stream is never used to fit/update
# either offline contextual router.
preds_1_train = model_agent1.predict(X1_train)
preds_2_train = model_agent2.predict(X2_train)

reward_train_1 = (preds_1_train == y_train).astype(int)
reward_train_2 = (preds_2_train == y_train).astype(int)

# --------------------------------------------------
# OFFLINE LOGISTIC ROUTER
# --------------------------------------------------
offline_log_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
offline_log_1.fit(Xc_train, reward_train_1)

offline_log_2 = LogisticRegression(solver='lbfgs', max_iter=1000)
offline_log_2.fit(Xc_train, reward_train_2)

# --------------------------------------------------
# OFFLINE TREE ROUTER
# --------------------------------------------------
offline_tree_1 = RandomForestRegressor(
    n_estimators=20,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)
offline_tree_1.fit(Xc_train, reward_train_1)

offline_tree_2 = RandomForestRegressor(
    n_estimators=20,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42
)
offline_tree_2.fit(Xc_train, reward_train_2)

print("    > Offline contextual routers trained on the training split only.")


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
# PLUG-IN QUANTILE-THRESHOLD POLICIES (new code)
# ============================================================
# Prop. 2.1: the optimal constrained policy gives agent index 0 (the LR agent, capacity
# `capacity_agent0`) the cap0-share of tasks with the largest reward gap
#     delta(x) = mu_0(x) - mu_1(x),
# i.e. it thresholds delta at the (1 - cap0) quantile of its distribution.
# A plug-in policy estimates mu_0, mu_1 and plugs the estimates into this rule.
#
# Offline plug-in: mu-hat = the train-only offline routers above (no online updates).
#     The cutoff is set once from the delta-hat distribution on the TRAINING contexts.
# Online plug-in:  mu-hat = the same online models as our policies (logistic Laplace model /
#     bootstrapped trees), using their point (greedy) estimates and updated from bandit
#     feedback. The cutoff is the running (1 - cap0) quantile of the delta-hat values of all
#     tasks seen so far (each value as computed when its task arrived). No queue is used.
#
# Ties at the cutoff (common for tree predictions) are broken at random with the probability
# that makes the reference distribution hit cap0 exactly (stochastic tie-breaking at boundary
# contexts, as in Sec. 2.3).

import os


def threshold_rule(delta_ref, cap0):
    """Return (tau, q): assign agent 0 if delta > tau; if delta == tau, assign agent 0 with
    probability q. Chosen so that exactly a cap0 fraction of delta_ref goes to agent 0."""
    delta_ref = np.asarray(delta_ref, dtype=float)
    n = len(delta_ref)
    k = cap0 * n
    if k <= 0:
        return np.inf, 0.0
    if k >= n:
        return -np.inf, 1.0
    kc = int(np.ceil(k))
    tau = np.partition(delta_ref, n - kc)[n - kc]          # kc-th largest value
    n_above = np.count_nonzero(delta_ref > tau)
    n_equal = np.count_nonzero(delta_ref == tau)
    q = float(np.clip((k - n_above) / n_equal, 0.0, 1.0))
    return tau, q


def offline_router_scores(X, kind):
    if kind == 'logistic':
        return offline_log_1.predict_proba(X)[:, 1], offline_log_2.predict_proba(X)[:, 1]
    return offline_tree_1.predict(X), offline_tree_2.predict(X)


# Offline delta-hat on training contexts (sets the cutoff) and on the test stream.
offline_delta = {}
for kind in ['logistic', 'tree']:
    m0_tr, m1_tr = offline_router_scores(Xc_train, kind)
    m0_te, m1_te = offline_router_scores(Xc_test, kind)
    offline_delta[kind] = {'train': m0_tr - m1_tr, 'test': m0_te - m1_te}


def run_offline_plugin(cap0, perm, kind, rng_tie):
    """Fixed offline estimates, fixed cutoff from the training split."""
    tau, q = threshold_rule(offline_delta[kind]['train'], cap0)
    delta = offline_delta[kind]['test'][perm]
    rewards = rewards_matrix_test[perm]
    u = rng_tie.random(len(delta))
    actions = np.where((delta > tau) | ((delta == tau) & (u < q)), 0, 1)
    step_rewards = rewards[np.arange(len(actions)), actions]
    return step_rewards.mean(), step_rewards, actions, tau, q


def run_online_plugin(cap0, perm, kind, rng_tie):
    """Online estimates (greedy point estimates, updated from bandit feedback) with a
    running-quantile cutoff. Same model classes and settings as run_simulation."""
    Xs = Xc_test[perm]
    Rs = rewards_matrix_test[perm]
    n, d = Xs.shape
    rng_model = np.random.default_rng(42)                 # as in run_simulation
    if kind == 'tree':
        agents = [AgentPosteriorRewardTreeTS(rng=rng_model) for _ in range(2)]
        predict = lambda a, x: a.predict_reward(x)
    else:
        agents = [ContextualLogRegAgent(d, alpha=0.5) for _ in range(2)]
        predict = lambda a, x: a.predict_reward_prob(x)

    hist = np.empty(n)
    step_rewards = np.zeros(n)
    actions = np.zeros(n, dtype=int)
    for t in range(n):
        x = Xs[t]
        delta = predict(agents[0], x) - predict(agents[1], x)
        hist[t] = delta
        tau, q = threshold_rule(hist[:t + 1], cap0)
        a = 0 if (delta > tau or (delta == tau and rng_tie.random() < q)) else 1
        r = Rs[t, a]
        agents[a].update(x, r)
        actions[t] = a
        step_rewards[t] = r
    return step_rewards.mean(), step_rewards, actions


# ============================================================
# EXPERIMENT LOOP: 10 RUNS, CHECKPOINT SAVES AFTER RUN 5 AND RUN 10
# ============================================================
capacities = np.array([0.2, 0.4, 0.5, 0.6, 0.8])           # capacity of agent index 0 (LR)
n_runs_per_point = 10
SAVE_AT = {5, 10}                                         # write results after these runs

OUT_DIR = f"plugin_{n_runs_per_point}_run"
os.makedirs(OUT_DIR, exist_ok=True)

POLICIES = [
    ("Offline plug-in Logistic", "offline", "logistic"),
    ("Offline plug-in Tree",     "offline", "tree"),
    ("Online plug-in Logistic",  "online",  "logistic"),
    ("Online plug-in Tree",      "online",  "tree"),
]

run_rows = []
offline_cutoffs = {}
step_rewards_accum = {p[0]: {float(c): [] for c in capacities} for p in POLICIES}
step_actions_accum = {p[0]: {float(c): [] for c in capacities} for p in POLICIES}


def save_results(n_done):
    """Write all results for runs 1..n_done. File names carry runs{n_done}, so the run-5
    checkpoint is kept and the run-10 files are written alongside it."""
    tag = f"banking_seed_{seed}_runs{n_done}_plugin"
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(f"{OUT_DIR}/{tag}_raw_runs.csv", index=False)

    summary_df = (
        run_df.groupby(["capacity_agent0", "capacity_agent1", "policy"], as_index=False)
        .agg(n_runs=("error_rate", "count"),
             mean_error=("error_rate", "mean"),
             sd_error=("error_rate", "std"),
             se_error=("error_rate", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
             mean_realized_frac_agent0=("realized_frac_agent0", "mean"))
    )
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
    with open(f"{OUT_DIR}/{tag}_offline_cutoffs.json", "w") as f:
        json.dump(offline_cutoffs, f, indent=2)

    np.save(f"{OUT_DIR}/{tag}_step_rewards.npy",
            {k: {c: np.stack(v) for c, v in d.items()} for k, d in step_rewards_accum.items()})
    np.save(f"{OUT_DIR}/{tag}_step_actions.npy",
            {k: {c: np.stack(v) for c, v in d.items()} for k, d in step_actions_accum.items()})

    print(f"\n--- Saved results for runs 1-{n_done} to {OUT_DIR}/{tag}_* ---")
    print(summary_df.round(4).to_string(index=False), flush=True)


print("Running plug-in quantile-threshold policies...")
for r in range(1, n_runs_per_point + 1):
    print(f"\n=== Plug-in run {r}/{n_runs_per_point} ===", flush=True)

    # Same permutation construction as the online and offline-router scripts.
    rng_run = np.random.default_rng(seed + r)
    perm_r = rng_run.permutation(len(Xc_test))

    for ci, cap in enumerate(capacities):
        for pi, (name, stage, kind) in enumerate(POLICIES):
            rng_tie = np.random.default_rng([seed, r, ci, pi])   # tie-breaks only
            if stage == "offline":
                avg, steps, acts, tau, q = run_offline_plugin(cap, perm_r, kind, rng_tie)
                offline_cutoffs.setdefault(name, {})[float(cap)] = {"tau": float(tau), "q": q}
            else:
                avg, steps, acts = run_online_plugin(cap, perm_r, kind, rng_tie)

            frac0 = float(np.mean(acts == 0))
            run_rows.append({
                "seed": seed, "run": r,
                "capacity_agent0": float(cap), "capacity_agent1": float(1.0 - cap),
                "policy": name, "avg_reward": avg, "error_rate": 1.0 - avg,
                "realized_frac_agent0": frac0,
            })
            step_rewards_accum[name][float(cap)].append(steps)
            step_actions_accum[name][float(cap)].append(acts)
            print(f"  cap0={cap:.1f} {name:26s} error={1.0 - avg:.4f} "
                  f"realized frac agent0={frac0:.4f}", flush=True)

    if r in SAVE_AT:
        save_results(r)

print("\nFinished.")
