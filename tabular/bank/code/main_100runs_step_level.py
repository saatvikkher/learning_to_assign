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
import os  # added: create output folders used below
os.makedirs(f"new_final", exist_ok=True)
os.makedirs(f"two_agents_100_run", exist_ok=True)
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

# --------------------------------------------------
# REWARD TARGETS (EXPERT CORRECTNESS)
# --------------------------------------------------
preds_1_test = model_agent1.predict(X1_test)
preds_2_test = model_agent2.predict(X2_test)

reward_target_1 = (preds_1_test == y_test).astype(int)
reward_target_2 = (preds_2_test == y_test).astype(int)
acc_agent1 = reward_target_1.mean()
acc_agent2 = reward_target_2.mean()
print('agent 1 marginal accuracy: ', acc_agent1)
print('agent 2 marginal accuracy: ', acc_agent2)

# Save the feature split, number of online tasks T and each agent's standalone accuracy on the T tasks
# (used for Table 1, the non-contextual baseline and the endpoints of the capacity plots).
import os as _os, json as _json
_os.makedirs("two_agents_100_run", exist_ok=True)
with open(f"two_agents_100_run/banking_seed_{seed}_agents.json", "w") as _f:
    _json.dump({
        "seed": seed,
        "n_tasks_T": int(Xc_test.shape[0]),
        "agent1_logistic_regression": {"features": list(agent1_cat) + list(agent1_cont), "accuracy": float(acc_agent1)},
        "agent2_xgboost": {"features": list(agent2_cat) + list(agent2_cont), "accuracy": float(acc_agent2)},
        "note": "code agent 1 = logistic regression = paper Agent 2; code agent 2 = XGBoost = paper Agent 1",
    }, _f, indent=2)

# --------------------------------------------------
# OFFLINE LOGISTIC POLICIES (ON CONTEXT)
# --------------------------------------------------
offline_log_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
offline_log_1.fit(Xc_test, reward_target_1)

offline_log_2 = LogisticRegression(solver='lbfgs', max_iter=1000)
offline_log_2.fit(Xc_test, reward_target_2)

# --------------------------------------------------
# OFFLINE TREE POLICIES (ON CONTEXT)
# --------------------------------------------------
offline_tree_1 = RandomForestRegressor(
    n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=42
)
offline_tree_1.fit(Xc_test, reward_target_1)

offline_tree_2 = RandomForestRegressor(
    n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=42
)
offline_tree_2.fit(Xc_test, reward_target_2)

print("    > Expert Models & Offline Policies Trained.")


# ==========================================
# 3. TEST ENVIRONMENT
# ==========================================

preds_1_test = model_agent1.predict(X1_test)
preds_2_test = model_agent2.predict(X2_test)

rewards_matrix_test = np.column_stack([
    (preds_1_test == y_test).astype(int),
    (preds_2_test == y_test).astype(int)
])

# ==========================================
# 3. SIMULATION
# ==========================================
def run_offline_only(
    X,
    rewards_matrix,
    p_targets,
    eta=1.0,
    mode='offline_logistic',
    seed=42
):
    """
    Evaluate offline policies only.
    - Single run
    - No learning / no updates
    - Optional capacity constraint via virtual queue
    """

    rng = np.random.default_rng(seed)

    n_steps, dim = X.shape
    n_k = len(p_targets)

    perm = rng.permutation(n_steps)
    # X_shuffled = X[perm]
    # rewards_shuffled = rewards_matrix[perm]
    # see the same x, only run once
    X_shuffled = X
    rewards_shuffled = rewards_matrix

    Q = np.zeros(n_k)
    cumulative_reward = 0.0

    for t in range(n_steps):
        x_t = X_shuffled[t]
        scores = np.zeros(n_k)

        # ------------------------------
        # SCORING (OFFLINE ONLY)
        # ------------------------------
        if mode == 'offline_logistic':
            scores[0] = offline_log_1.predict_proba(x_t.reshape(1, -1))[0, 1] - eta * Q[0]
            scores[1] = offline_log_2.predict_proba(x_t.reshape(1, -1))[0, 1] - eta * Q[1]

        elif mode == 'offline_tree':
            scores[0] = offline_tree_1.predict(x_t.reshape(1, -1))[0] - eta * Q[0]
            scores[1] = offline_tree_2.predict(x_t.reshape(1, -1))[0] - eta * Q[1]

        elif mode == 'offline_logistic_unconstrained':
            scores[0] = offline_log_1.predict_proba(x_t.reshape(1, -1))[0, 1]
            scores[1] = offline_log_2.predict_proba(x_t.reshape(1, -1))[0, 1]

        elif mode == 'offline_tree_unconstrained':
            scores[0] = offline_tree_1.predict(x_t.reshape(1, -1))[0]
            scores[1] = offline_tree_2.predict(x_t.reshape(1, -1))[0]

        else:
            raise ValueError(f"Unsupported offline mode: {mode}")

        # ------------------------------
        # ACTION SELECTION
        # ------------------------------
        action = np.argmax(scores)

        reward = rewards_shuffled[t, action]
        cumulative_reward += reward
        # ------------------------------
        # QUEUE UPDATE (ONLY IF CONSTRAINED)
        # ------------------------------
        if 'unconstrained' not in mode:
            z = np.zeros(n_k)
            z[action] = 1.0
            Q = np.maximum(Q + z - p_targets, 0.0)

    return cumulative_reward / n_steps


def run_simulation(X, rewards_matrix, p_targets, perm, # <-- ADD
                   n_runs=100, eta=1.0, mode='proposed'):
    n_steps, dim = X.shape
    run_avg_rewards = []
    all_step_rewards = []        
    all_step_actions = []     
    rng = np.random.default_rng(42)

    for r in range(n_runs):
        # perm = np.random.permutation(n_steps)
        X_shuffled = X[perm]
        rewards_shuffled = rewards_matrix[perm]
        
        n_k = len(p_targets)
        Q = np.zeros(n_k) 
        
        if mode in ['tree_ts', 'tree_greedy']:
            agents = [AgentPosteriorRewardTreeTS(rng=rng) for _ in range(n_k)]
        else:
            agents = [ContextualLogRegAgent(dim, alpha=0.5) for _ in range(n_k)]
            
        cumulative_reward = 0
        step_rewards = np.zeros(n_steps)   # NEW
        step_actions = np.zeros(n_steps, dtype=int)   # <-- ADD
        
        for t in range(n_steps):
            x_t = X_shuffled[t]
            scores = np.zeros(n_k)
            
            # --- SCORING ---
            if mode == 'random':
                pass
                
            elif mode == 'tree_ts':
                for i in range(n_k):
                    val = agents[i].sample_reward(x_t)
                    val -= eta * Q[i]
                    scores[i] = val
            elif mode == 'tree_greedy':
                for i in range(n_k):
                    val = agents[i].predict_reward(x_t)
                    val -= eta * Q[i]
                    scores[i] = val
            elif mode == 'log_ts':
                for i in range(n_k):
                    val = agents[i].sample_prob(x_t)
                    val -= eta * Q[i]
                    scores[i] = val

            elif mode == 'log_greedy':
                for i in range(n_k):
                    val = agents[i].predict_reward_prob(x_t)
                    val -= eta * Q[i]
                    scores[i] = val
            
                    
            # else: # Logistic TS or UCB
            #     for i in range(n_k):
            #         if 'ucb' in mode: val = agents[i].get_ucb_prob(x_t)
            #         else: val = agents[i].sample_prob(x_t)
            #         val -= eta * Q[i]
            #         scores[i] = val
            
            # --- SELECTION ---
            if mode == 'random':
                action = np.random.choice(n_k, p=p_targets)
            else:
                action = np.argmax(scores)
                
            step_actions[t] = action   # <-- ADD
            reward = rewards_shuffled[t, action]
            cumulative_reward += reward
            step_rewards[t] = reward  
            
            # --- UPDATE ---
            if mode not in ['random']:
                agents[action].update(x_t, reward)
            
            z = np.zeros(n_k); z[action] = 1.0
            Q = np.maximum(Q + z - p_targets, 0.0)
              
            
        run_avg_rewards.append(cumulative_reward / n_steps)
        all_step_rewards.append(step_rewards)    # NEW
        all_step_actions.append(step_actions)   # <-- ADD
    return np.mean(run_avg_rewards), np.array(all_step_rewards), np.array(all_step_actions)   # <-- ADD)

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

def plot_results(results, capacities, n_runs, seed, suffix):
    plt.figure(figsize=(10, 6))

    vc = 1 - capacities

    if 'Offline Tree Un' in results:
        plt.axhline(
            y=results['Offline Tree Un'][0],
            color='darkred',
            linestyle='-.',
            linewidth=2,
            label='Offline Tree (Unconstrained)'
        )
    if 'Offline Logistic Un' in results:        
        plt.axhline(
            y=results['Offline Logistic Un'][0],
            color='black',
            linestyle=':',
            linewidth=2,
            label='Offline Logistic (Unconstrained)'
        )

    plt.plot(vc, results['Offline Tree'], label='Offline Tree', marker='o')
    plt.plot(vc, results['Offline Logistic'], label='Offline Logistic', marker='o')
    plt.plot(vc, results['Tree TS'], label='Tree TS', marker='s')
    plt.plot(vc, results['Tree greedy'], label='Tree greedy', marker='s')
    plt.plot(vc, results['Logistic TS'], label='Logistic TS', marker='^')
    plt.plot(vc, results['Logistic greedy'], label='Logistic greedy', marker='^')

    plt.title(f'Error vs Capacity (first {n_runs} runs)')
    plt.xlabel('Capacity of Arm 0')
    plt.ylabel('Average Error')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        f"new_final/banking_seed_{seed}_{suffix}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ==========================================
# 6. EXPERIMENT LOOP
# ==========================================
# capacities = np.linspace(0.0, 1.0, 5)
# ==========================================
# 6. EXPERIMENT LOOP WITH SNAPSHOTS
# ==========================================

capacities = np.array([0.2, 0.4, 0.5, 0.6, 0.8])
n_runs_per_point = 100
BATCH = 5
eta_val = 0.5

# ------------------------------------------
# 1. OFFLINE RESULTS (COMPUTE ONCE)
# ------------------------------------------
results_offline = {
    'Offline Logistic': [],
    'Offline Tree': [],
    'Offline Logistic Un': [],
    'Offline Tree Un': []
}

print("Running offline baselines...")

for cap in capacities:
    if cap > 1.01:
        continue
    targets = np.array([cap, 1.0 - cap])

    avg = run_offline_only(
        Xc_test, rewards_matrix_test,
        p_targets=targets, eta=eta_val,
        mode='offline_logistic'
    )
    results_offline['Offline Logistic'].append(1.0 - avg)

    avg = run_offline_only(
        Xc_test, rewards_matrix_test,
        p_targets=targets, eta=eta_val,
        mode='offline_tree'
    )
    results_offline['Offline Tree'].append(1.0 - avg)

# unconstrained (scalar → repeat for alignment)
avg = run_offline_only(
    Xc_test, rewards_matrix_test,
    p_targets=np.array([0.0, 0.0]),
    eta=eta_val, mode='offline_tree_unconstrained'
)
results_offline['Offline Tree Un'] = [1.0 - avg] * len(capacities)

avg = run_offline_only(
    Xc_test, rewards_matrix_test,
    p_targets=np.array([0.0, 0.0]),
    eta=eta_val, mode='offline_logistic_unconstrained'
)
results_offline['Offline Logistic Un'] = [1.0 - avg] * len(capacities)

# ------------------------------------------
# 2. ONLINE ACCUMULATORS (GLOBAL)
# ------------------------------------------
online_stats = {
    'Logistic TS': {cap: [] for cap in capacities},
    'Tree TS':     {cap: [] for cap in capacities},
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy':     {cap: [] for cap in capacities}
}

step_rewards_accum = {
    'Logistic TS': {cap: [] for cap in capacities},
    'Tree TS':     {cap: [] for cap in capacities},
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy':     {cap: [] for cap in capacities}
}
step_actions_accum = {
    'Logistic TS': {cap: [] for cap in capacities},
    'Tree TS':     {cap: [] for cap in capacities},
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy':     {cap: [] for cap in capacities}
}

# ------------------------------------------
# 3. GLOBAL ONLINE RUN LOOP
# ------------------------------------------
print("Running online policies with snapshots...")
# perm_store = []
# X_store = []
# rewards_store = []

for r in range(1, n_runs_per_point + 1):
    print(f"\n=== Global online run {r}/{n_runs_per_point} ===")

    rng_run = np.random.default_rng(seed + r)
    perm_r = rng_run.permutation(len(Xc_test))   # <-- ONE perm per run

    
    for cap in capacities:
        if cap > 1.01:
            continue

        targets = np.array([cap, 1.0 - cap])

        # ----- Logistic TS -----
        avg, steps, actions = run_simulation(
            Xc_test,
            rewards_matrix_test,
            targets,
            perm=perm_r,          # <-- SAME perm
            n_runs=1,
            eta=eta_val,
            mode='log_ts'
        )
        online_stats['Logistic TS'][cap].append(avg)
        step_rewards_accum['Logistic TS'][cap].append(steps)
        step_actions_accum['Logistic TS'][cap].append(actions)

        # Logistics Greedy 
        avg, steps, actions = run_simulation(
            Xc_test,
            rewards_matrix_test,
            targets,
            perm=perm_r,          # <-- SAME perm
            n_runs=1,
            eta=eta_val,
            mode='log_greedy'
        )
        online_stats['Logistic greedy'][cap].append(avg)
        step_rewards_accum['Logistic greedy'][cap].append(steps)
        step_actions_accum['Logistic greedy'][cap].append(actions)

        
        # ----- Tree TS -----
        avg, steps, actions = run_simulation(
            Xc_test,
            rewards_matrix_test,
            targets,
            perm=perm_r,          # <-- SAME perm
            n_runs=1,
            eta=eta_val,
            mode='tree_ts'
        )
        online_stats['Tree TS'][cap].append(avg)
        step_rewards_accum['Tree TS'][cap].append(steps)
        step_actions_accum['Tree TS'][cap].append(actions)

        # ----- Tree TS -----
        avg, steps, actions = run_simulation(
            Xc_test,
            rewards_matrix_test,
            targets,
            perm=perm_r,          # <-- SAME perm
            n_runs=1,
            eta=eta_val,
            mode='tree_greedy'
        )
        online_stats['Tree greedy'][cap].append(avg)
        step_rewards_accum['Tree greedy'][cap].append(steps)
        step_actions_accum['Tree greedy'][cap].append(actions)

    # --------------------------------------
    # SNAPSHOT EVERY 5 RUNS
    # --------------------------------------
    if (r >= 50) and (r % BATCH == 0):
        print(f"Saving snapshot at run {r}")

        # ----- Average reward snapshot -----
        results_snapshot = {
            **results_offline,
            'Logistic TS': [],
            'Logistic greedy': [],
            'Tree TS': [],
            'Tree greedy': []
        }

        for cap in capacities:
            results_snapshot['Logistic TS'].append(
                1.0 - np.mean(online_stats['Logistic TS'][cap])
            )
            results_snapshot['Logistic greedy'].append(
                1.0 - np.mean(online_stats['Logistic greedy'][cap])
            )
            results_snapshot['Tree TS'].append(
                1.0 - np.mean(online_stats['Tree TS'][cap])
            )
            results_snapshot['Tree greedy'].append(
                1.0 - np.mean(online_stats['Tree greedy'][cap])
            )

        # ----- Step-reward snapshot -----
        step_rewards_snapshot = {
            'Logistic TS': {
                cap: np.concatenate(step_rewards_accum['Logistic TS'][cap], axis=0)
                for cap in capacities
            },
            'Tree TS': {
                cap: np.concatenate(step_rewards_accum['Tree TS'][cap], axis=0)
                for cap in capacities
            },
            'Logistic greedy': {
                cap: np.concatenate(step_rewards_accum['Logistic greedy'][cap], axis=0)
                for cap in capacities
            },
            'Tree greedy': {
                cap: np.concatenate(step_rewards_accum['Tree greedy'][cap], axis=0)
                for cap in capacities
            }
        }

        # step-action snapshot 
        step_actions_snapshot = {
            'Logistic TS': {
                cap: np.concatenate(step_actions_accum['Logistic TS'][cap], axis=0)
                for cap in capacities
            },
            'Tree TS': {
                cap: np.concatenate(step_actions_accum['Tree TS'][cap], axis=0)
                for cap in capacities
            },
            'Logistic greedy': {
                cap: np.concatenate(step_actions_accum['Logistic greedy'][cap], axis=0)
                for cap in capacities
            },
            'Tree greedy': {
                cap: np.concatenate(step_actions_accum['Tree greedy'][cap], axis=0)
                for cap in capacities
            }
        }

        suffix = f"runs{r}"

        # ----- SAVE JSON -----
        with open(f"two_agents_100_run/banking_seed_{seed}_{suffix}.json", "w") as f:
            json.dump(results_snapshot, f, indent=2)

        # step_rewards saved separately (numpy-friendly)
        np.save(
            f"two_agents_100_run/banking_seed_{seed}_{suffix}_step_rewards.npy",
            step_rewards_snapshot
        )
        np.save(
            f"two_agents_100_run/banking_seed_{seed}_{suffix}_step_actions.npy",
            step_actions_snapshot
        )
