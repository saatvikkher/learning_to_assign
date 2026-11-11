import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ortools.graph.python import min_cost_flow

# ============================================================
# 0. SETUP
# ============================================================
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rng = np.random.default_rng(seed)
np.random.seed(seed)

print(f"\n=== Running batch queue-penalized MCMF with seed = {seed} ===\n")

# ===== CHANGE HERE =====
# Main experiment knobs
capacities = np.array([0.2, 0.4, 0.5, 0.6, 0.8])   # target share of agent 1
batch_size = 100                                     # can set 99, 100, etc.
n_runs_per_point = 50
eta_val = 0.5                                        # queue penalty strength
output_dir = f"batch_queue_results"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 1. FEATURE DEFINITIONS
# ============================================================
categorical_features = [
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'contact',
    'month',
    'poutcome',
    'day_of_week'
]

continuous_features = [
    'age',
    'duration',
    'campaign',
    'pdays',
    'previous'
]

def split_features_randomly(cat_features, cont_features, rng, frac=0.2):
    all_features = cat_features + cont_features
    n_total = len(all_features)
    n_agent1 = max(1, int(np.floor(frac * n_total)))

    agent1_feats = rng.choice(all_features, size=n_agent1, replace=False).tolist()
    agent2_feats = [f for f in all_features if f not in agent1_feats]

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

# ============================================================
# 2. DATA LOADING & PREP
# ============================================================
def load_and_prep_data(agent1_cat, agent1_cont, agent2_cat, agent2_cont):
    df = pd.read_csv('../data/bank_marketing_raw.csv', sep=';')
    df["target"] = (df["y"] == "yes").astype(int)
    n = len(df)

    # ---------- Agent 1 ----------
    if len(agent1_cat) > 0:
        X1_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df[agent1_cat])
    else:
        X1_cat = np.empty((n, 0))

    if len(agent1_cont) > 0:
        X1_cont = StandardScaler().fit_transform(df[agent1_cont])
    else:
        X1_cont = np.empty((n, 0))

    X1 = np.hstack([X1_cat, X1_cont])

    # ---------- Agent 2 ----------
    if len(agent2_cat) > 0:
        X2_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df[agent2_cat])
    else:
        X2_cat = np.empty((n, 0))

    if len(agent2_cont) > 0:
        X2_cont = StandardScaler().fit_transform(df[agent2_cont])
    else:
        X2_cont = np.empty((n, 0))

    X2 = np.hstack([X2_cat, X2_cont])

    # ---------- Full context ----------
    Xc_cat = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df[context_cat])
    Xc_cont = StandardScaler().fit_transform(df[context_cont])
    Xc = np.hstack([Xc_cat, Xc_cont])

    return train_test_split(
        Xc, X1, X2, df["target"].values,
        test_size=0.3, random_state=42
    )

Xc_train, Xc_test, X1_train, X1_test, X2_train, X2_test, y_train, y_test = \
    load_and_prep_data(agent1_cat, agent1_cont, agent2_cat, agent2_cont)

# ============================================================
# 3. TRAIN TRUE EXPERT MODELS
# These generate the "environment" reward matrix.
# ============================================================
print("\n--- Step 3: Training expert models & offline reward estimators ---")

# Agent models: code agent 1 = logistic regression (paper Agent 2),
#               code agent 2 = XGBoost (paper Agent 1).
model_agent1 = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
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

preds_1_test = model_agent1.predict(X1_test)
preds_2_test = model_agent2.predict(X2_test)

reward_target_1 = (preds_1_test == y_test).astype(int)
reward_target_2 = (preds_2_test == y_test).astype(int)

acc_agent1 = reward_target_1.mean()
acc_agent2 = reward_target_2.mean()
print("agent 1 marginal accuracy:", acc_agent1)
print("agent 2 marginal accuracy:", acc_agent2)

# Offline contextual reward estimators: E[reward_i | x]
offline_log_1 = LogisticRegression(solver='lbfgs', max_iter=1000)
offline_log_1.fit(Xc_test, reward_target_1)

offline_log_2 = LogisticRegression(solver='lbfgs', max_iter=1000)
offline_log_2.fit(Xc_test, reward_target_2)

offline_tree_1 = RandomForestRegressor(
    n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=42
)
offline_tree_1.fit(Xc_test, reward_target_1)

offline_tree_2 = RandomForestRegressor(
    n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=42
)
offline_tree_2.fit(Xc_test, reward_target_2)

print("Trained offline reward estimators.")

# ============================================================
# 4. TEST ENVIRONMENT
# rewards_matrix_test[t, i] is realized reward if task t goes to agent i
# ============================================================
rewards_matrix_test = np.column_stack([
    (preds_1_test == y_test).astype(int),
    (preds_2_test == y_test).astype(int)
])

# ============================================================
# 5. ONLINE LEARNER CLASSES
# Same idea as your sequential learner, but updates happen after
# assignment within each batch.
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
        w = p_hat * (1 - p_hat)
        w = np.clip(w, 1e-4, 0.25)

        Ax = self.A_inv @ x
        numerator = w * np.outer(Ax, Ax)
        denominator = 1 + w * np.dot(x, Ax)
        self.A_inv = self.A_inv - numerator / denominator
        self.A_inv = (self.A_inv + self.A_inv.T) / 2.0

        g = (reward - p_hat) * x
        self.m = self.m + self.A_inv @ g

class AgentPosteriorRewardTreeTS:
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
            return 0.5
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
# 6. MCMF ASSIGNMENT, CHANGED HERE
# This is the batch solver. Here scores already include
# queue penalties: score_ij = predicted_reward_ij - eta * Q_i
# ============================================================
def assign_mcmf_scores(scores, caps):
    """
    scores: np.array of shape (n_agents, batch_size)
            larger is better
    caps: integer capacities per agent for THIS batch
          must sum to batch_size

    returns:
        assignment: array of shape (batch_size,), assignment[j] = i
        total_score: sum of selected scores
    """
    n_agents, n_tasks = scores.shape
    if sum(caps) != n_tasks:
        raise ValueError("For exact assignment, sum(caps) must equal number of tasks in the batch.")

    src, sink = 0, 1
    task_off = 2
    agent_off = 2 + n_tasks

    smcf = min_cost_flow.SimpleMinCostFlow()

    # source -> tasks
    for j in range(n_tasks):
        smcf.add_arc_with_capacity_and_unit_cost(src, task_off + j, 1, 0)

    # task -> agent
    scale = 10**6
    for i in range(n_agents):
        for j in range(n_tasks):
            cost = -int(np.round(scores[i, j] * scale))
            smcf.add_arc_with_capacity_and_unit_cost(task_off + j, agent_off + i, 1, cost)

    # agent -> sink
    for i in range(n_agents):
        smcf.add_arc_with_capacity_and_unit_cost(agent_off + i, sink, int(caps[i]), 0)

    smcf.set_node_supply(src, n_tasks)
    smcf.set_node_supply(sink, -n_tasks)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError("Min-cost flow not optimal")

    assignment = np.full(n_tasks, -1, dtype=int)
    for a in range(smcf.num_arcs()):
        if smcf.flow(a) > 0:
            u = smcf.tail(a)
            v = smcf.head(a)
            if task_off <= u < agent_off and v >= agent_off:
                j = u - task_off
                i = v - agent_off
                assignment[j] = i

    if np.any(assignment < 0):
        raise RuntimeError("Some tasks were not assigned.")

    total_score = -smcf.optimal_cost() / scale
    return assignment, total_score

# ============================================================
# 7. BATCH CAPACITY RULE FROM QUEUE
# IMPORTANT:
# We still solve an exact integer assignment each batch, so we need
# integer caps. The queue tells us how much to bias these caps.
#
# Rule:
# desired_i = B * p_i - Q_i
# then project to integers summing to B via floor + largest remainder
#
# If Q_i is large, desired_i shrinks, so that agent gets fewer tasks
# in the next batch.
# ============================================================
def batch_caps_from_queue(batch_size, p_targets, Q):
    """
    Returns integer batch capacities that sum exactly to batch_size.

    desired_i = batch_size * p_i - Q_i

    Then:
    1. clip small negatives to 0
    2. renormalize to sum to batch_size
    3. use largest-remainder integer rounding
    """
    desired = batch_size * np.asarray(p_targets) - np.asarray(Q)
    desired = np.maximum(desired, 0.0)

    if desired.sum() <= 0:
        desired = np.ones_like(desired) / len(desired) * batch_size
    else:
        desired = desired / desired.sum() * batch_size

    base = np.floor(desired).astype(int)
    remainder = desired - base
    shortfall = batch_size - base.sum()

    if shortfall > 0:
        idx = np.argsort(-remainder)[:shortfall]
        base[idx] += 1

    return base

# ============================================================
# 8. BATCH SIMULATION WITH QUEUE-PENALIZED MCMF
# This is the main function you want.
# ============================================================
def run_batch_queue_simulation(
    X,
    rewards_matrix,
    p_targets,
    perm,
    batch_size=100,
    n_runs=1,
    eta=1.0,
    mode='log_ts',
    seed=42
):
    n_steps, dim = X.shape
    n_agents = len(p_targets)
    rng_local = np.random.default_rng(seed)

    # ===== CHANGE HERE =====
    # no truncation anymore
    n_batches = int(np.ceil(n_steps / batch_size))

    run_avg_rewards = []
    all_step_rewards = []
    all_step_actions = []
    all_batch_queues = []
    all_batch_counts = []

    for r in range(n_runs):
        X_shuffled = X[perm]
        rewards_shuffled = rewards_matrix[perm]

        Q = np.zeros(n_agents)

        if mode in ['tree_ts', 'tree_greedy']:
            agents = [AgentPosteriorRewardTreeTS(rng=rng_local) for _ in range(n_agents)]
        else:
            agents = [ContextualLogRegAgent(dim, alpha=0.5) for _ in range(n_agents)]

        cumulative_reward = 0.0
        step_rewards = np.zeros(n_steps)
        step_actions = np.zeros(n_steps, dtype=int)
        batch_queues = []
        batch_counts = []

        for b in range(n_batches):
            # ===== CHANGE HERE =====
            start = b * batch_size
            end = min(start + batch_size, n_steps)
            current_batch_size = end - start

            X_batch = X_shuffled[start:end]
            R_batch = rewards_shuffled[start:end]

            pred_scores = np.zeros((n_agents, current_batch_size))   # ===== CHANGE HERE =====

            for j in range(current_batch_size):   # ===== CHANGE HERE =====
                x_j = X_batch[j]

                for i in range(n_agents):
                    if mode == 'log_ts':
                        pred_scores[i, j] = agents[i].sample_prob(x_j)
                    elif mode == 'log_greedy':
                        pred_scores[i, j] = agents[i].predict_reward_prob(x_j)
                    elif mode == 'tree_ts':
                        pred_scores[i, j] = agents[i].sample_reward(x_j)
                    elif mode == 'tree_greedy':
                        pred_scores[i, j] = agents[i].predict_reward(x_j)
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")

            penalized_scores = pred_scores - eta * Q.reshape(-1, 1)

            # ===== CHANGE HERE =====
            caps_batch = batch_caps_from_queue(current_batch_size, p_targets, Q)

            if caps_batch.sum() != current_batch_size:   # ===== CHANGE HERE =====
                raise RuntimeError("caps_batch does not sum to current_batch_size")

            assignment, _ = assign_mcmf_scores(penalized_scores, caps_batch)

            n_assigned = np.zeros(n_agents, dtype=int)

            for j in range(current_batch_size):   # ===== CHANGE HERE =====
                global_j = start + j
                a = assignment[j]
                reward = R_batch[j, a]

                step_actions[global_j] = a
                step_rewards[global_j] = reward
                cumulative_reward += reward
                n_assigned[a] += 1

                agents[a].update(X_batch[j], reward)

            # ===== CHANGE HERE =====
            Q = np.maximum(Q + n_assigned - current_batch_size * np.asarray(p_targets), 0.0)

            batch_queues.append(Q.copy())
            batch_counts.append(n_assigned.copy())

        run_avg_rewards.append(cumulative_reward / n_steps)
        all_step_rewards.append(step_rewards)
        all_step_actions.append(step_actions)
        all_batch_queues.append(np.array(batch_queues))
        all_batch_counts.append(np.array(batch_counts))

    return (
        np.mean(run_avg_rewards),
        np.array(all_step_rewards),
        np.array(all_step_actions),
        np.array(all_batch_queues),
        np.array(all_batch_counts)
    )

# ============================================================
# 9. OFFLINE BATCH BENCHMARKS
# These are useful comparators.
# ============================================================
def run_batch_offline_benchmark(
    X,
    rewards_matrix,
    p_targets,
    perm,
    batch_size=100,
    eta=1.0,
    mode='offline_logistic_queue'
):
    n_steps = X.shape[0]
    n_agents = len(p_targets)

    # ===== CHANGE HERE =====
    # no truncation anymore
    n_batches = int(np.ceil(n_steps / batch_size))

    X_shuffled = X[perm]
    rewards_shuffled = rewards_matrix[perm]

    Q = np.zeros(n_agents)
    cumulative_reward = 0.0
    step_actions = np.zeros(n_steps, dtype=int)
    batch_queues = []
    batch_counts = []

    for b in range(n_batches):
        # ===== CHANGE HERE =====
        start = b * batch_size
        end = min(start + batch_size, n_steps)
        current_batch_size = end - start

        X_batch = X_shuffled[start:end]
        R_batch = rewards_shuffled[start:end]

        pred_scores = np.zeros((n_agents, current_batch_size))   # ===== CHANGE HERE =====

        for j in range(current_batch_size):   # ===== CHANGE HERE =====
            x_j = X_batch[j].reshape(1, -1)

            if mode in ['offline_logistic_queue', 'offline_logistic_unconstrained']:
                pred_scores[0, j] = offline_log_1.predict_proba(x_j)[0, 1]
                pred_scores[1, j] = offline_log_2.predict_proba(x_j)[0, 1]
            elif mode in ['offline_tree_queue', 'offline_tree_unconstrained']:
                pred_scores[0, j] = offline_tree_1.predict(x_j)[0]
                pred_scores[1, j] = offline_tree_2.predict(x_j)[0]
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        if 'unconstrained' in mode:
            raw_caps = np.array([
                current_batch_size * p_targets[0],
                current_batch_size - current_batch_size * p_targets[0]
            ])   # ===== CHANGE HERE =====
            caps_batch = np.array([
                int(round(raw_caps[0])),
                current_batch_size - int(round(raw_caps[0]))
            ])   # ===== CHANGE HERE =====
            penalized_scores = pred_scores
        else:
            penalized_scores = pred_scores - eta * Q.reshape(-1, 1)
            caps_batch = batch_caps_from_queue(current_batch_size, p_targets, Q)   # ===== CHANGE HERE =====

        assignment, _ = assign_mcmf_scores(penalized_scores, caps_batch)

        n_assigned = np.zeros(n_agents, dtype=int)

        for j in range(current_batch_size):   # ===== CHANGE HERE =====
            global_j = start + j
            a = assignment[j]
            reward = R_batch[j, a]
            step_actions[global_j] = a
            cumulative_reward += reward
            n_assigned[a] += 1

        if 'unconstrained' not in mode:
            # ===== CHANGE HERE =====
            Q = np.maximum(Q + n_assigned - current_batch_size * np.asarray(p_targets), 0.0)

        batch_queues.append(Q.copy())
        batch_counts.append(n_assigned.copy())

    avg_reward = cumulative_reward / n_steps
    return avg_reward, step_actions, np.array(batch_queues), np.array(batch_counts)
# ============================================================
# 10. PLOTTING
# ============================================================
def plot_results(results, capacities, n_runs, seed, suffix, output_dir):
    plt.figure(figsize=(10, 6))
    vc = capacities

    if 'Offline Tree Un' in results:
        plt.axhline(
            y=results['Offline Tree Un'][0],
            linestyle='-.',
            linewidth=2,
            label='Offline Tree Unconstrained'
        )
    if 'Offline Logistic Un' in results:
        plt.axhline(
            y=results['Offline Logistic Un'][0],
            linestyle=':',
            linewidth=2,
            label='Offline Logistic Unconstrained'
        )

    plt.plot(vc, results['Offline Tree'], label='Offline Tree Queue-MCMF', marker='o')
    plt.plot(vc, results['Offline Logistic'], label='Offline Logistic Queue-MCMF', marker='o')
    plt.plot(vc, results['Tree TS'], label='Tree TS Queue-MCMF', marker='s')
    plt.plot(vc, results['Tree greedy'], label='Tree greedy Queue-MCMF', marker='s')
    plt.plot(vc, results['Logistic TS'], label='Logistic TS Queue-MCMF', marker='^')
    plt.plot(vc, results['Logistic greedy'], label='Logistic greedy Queue-MCMF', marker='^')

    plt.title(f'Error vs target share of agent 1 (batch size = {batch_size}, runs = {n_runs})')
    plt.xlabel('Target share of agent 1')
    plt.ylabel('Average error')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        os.path.join(output_dir, f"batch_queue_seed_{seed}_{suffix}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ============================================================
# 11. EXPERIMENT LOOP
# ============================================================
results_offline = {
    'Offline Logistic': [],
    'Offline Tree': [],
    'Offline Logistic Un': [],
    'Offline Tree Un': []
}

online_stats = {
    'Logistic TS': {cap: [] for cap in capacities},
    'Tree TS': {cap: [] for cap in capacities},
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy': {cap: [] for cap in capacities}
}

queue_logs = {
    'Logistic TS': {cap: [] for cap in capacities},
    'Tree TS': {cap: [] for cap in capacities},
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy': {cap: [] for cap in capacities},
    'Offline Logistic': {cap: [] for cap in capacities},
    'Offline Tree': {cap: [] for cap in capacities}
}

count_logs = {
    'Logistic TS': {cap: [] for cap in capacities},
    'Tree TS': {cap: [] for cap in capacities},
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy': {cap: [] for cap in capacities},
    'Offline Logistic': {cap: [] for cap in capacities},
    'Offline Tree': {cap: [] for cap in capacities}
}

print("Running experiments...")

for r in range(1, n_runs_per_point + 1):
    print(f"\n=== Run {r}/{n_runs_per_point} ===")
    rng_run = np.random.default_rng(seed + r)
    perm_r = rng_run.permutation(len(Xc_test))

    for cap in capacities:
        targets = np.array([cap, 1.0 - cap])

        # -----------------------------------
        # Offline queue-MCMF benchmark
        # -----------------------------------
        avg, _, q_hist, c_hist = run_batch_offline_benchmark(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, eta=eta_val, mode='offline_logistic_queue'
        )
        if r == 1:
            results_offline['Offline Logistic'].append(1.0 - avg)
        queue_logs['Offline Logistic'][cap].append(q_hist)
        count_logs['Offline Logistic'][cap].append(c_hist)

        avg, _, q_hist, c_hist = run_batch_offline_benchmark(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, eta=eta_val, mode='offline_tree_queue'
        )
        if r == 1:
            results_offline['Offline Tree'].append(1.0 - avg)
        queue_logs['Offline Tree'][cap].append(q_hist)
        count_logs['Offline Tree'][cap].append(c_hist)

        # -----------------------------------
        # Unconstrained offline lines
        # Only compute once
        # -----------------------------------
        if r == 1:
            avg_un, _, _, _ = run_batch_offline_benchmark(
                Xc_test, rewards_matrix_test, targets, perm_r,
                batch_size=batch_size, eta=eta_val, mode='offline_logistic_unconstrained'
            )
            results_offline['Offline Logistic Un'].append(1.0 - avg_un)

            avg_un, _, _, _ = run_batch_offline_benchmark(
                Xc_test, rewards_matrix_test, targets, perm_r,
                batch_size=batch_size, eta=eta_val, mode='offline_tree_unconstrained'
            )
            results_offline['Offline Tree Un'].append(1.0 - avg_un)

        # -----------------------------------
        # Online batch queue-MCMF policies
        # -----------------------------------
        avg, _, _, q_hist, c_hist = run_batch_queue_simulation(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, n_runs=1, eta=eta_val,
            mode='log_ts', seed=seed + 1000 + r
        )
        online_stats['Logistic TS'][cap].append(avg)
        queue_logs['Logistic TS'][cap].append(q_hist)
        count_logs['Logistic TS'][cap].append(c_hist)

        avg, _, _, q_hist, c_hist = run_batch_queue_simulation(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, n_runs=1, eta=eta_val,
            mode='log_greedy', seed=seed + 2000 + r
        )
        online_stats['Logistic greedy'][cap].append(avg)
        queue_logs['Logistic greedy'][cap].append(q_hist)
        count_logs['Logistic greedy'][cap].append(c_hist)

        avg, _, _, q_hist, c_hist = run_batch_queue_simulation(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, n_runs=1, eta=eta_val,
            mode='tree_ts', seed=seed + 3000 + r
        )
        online_stats['Tree TS'][cap].append(avg)
        queue_logs['Tree TS'][cap].append(q_hist)
        count_logs['Tree TS'][cap].append(c_hist)

        avg, _, _, q_hist, c_hist = run_batch_queue_simulation(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, n_runs=1, eta=eta_val,
            mode='tree_greedy', seed=seed + 4000 + r
        )
        online_stats['Tree greedy'][cap].append(avg)
        queue_logs['Tree greedy'][cap].append(q_hist)
        count_logs['Tree greedy'][cap].append(c_hist)

# ============================================================
# 12. COLLECT FINAL RESULTS
# ============================================================
results_final = {
    **results_offline,
    'Logistic TS': [],
    'Tree TS': [],
    'Logistic greedy': [],
    'Tree greedy': []
}

for cap in capacities:
    results_final['Logistic TS'].append(1.0 - np.mean(online_stats['Logistic TS'][cap]))
    results_final['Tree TS'].append(1.0 - np.mean(online_stats['Tree TS'][cap]))
    results_final['Logistic greedy'].append(1.0 - np.mean(online_stats['Logistic greedy'][cap]))
    results_final['Tree greedy'].append(1.0 - np.mean(online_stats['Tree greedy'][cap]))

# save json
with open(os.path.join(output_dir, f"batch_queue_seed_{seed}_final.json"), "w") as f:
    json.dump(results_final, f, indent=2)

# save queue/count logs
np.save(os.path.join(output_dir, f"batch_queue_seed_{seed}_queue_logs.npy"), queue_logs, allow_pickle=True)
np.save(os.path.join(output_dir, f"batch_queue_seed_{seed}_count_logs.npy"), count_logs, allow_pickle=True)

# plot
plot_results(
    results_final,
    capacities=capacities,
    n_runs=n_runs_per_point,
    seed=seed,
    suffix="final",
    output_dir=output_dir
)

print("\nDone.")
print("Saved results to:", output_dir)
print(json.dumps(results_final, indent=2))
