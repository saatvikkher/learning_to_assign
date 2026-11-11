import os
import sys
import json
import numpy as np
import pandas as pd

from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ortools.graph.python import min_cost_flow

# ============================================================
# 0. SETUP
# ============================================================
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100

rng = np.random.default_rng(seed)
np.random.seed(seed)

print(f"\n=== Running batch queue-penalized MCMF with seed = {seed} ===\n")

# ===== MODIFIED =====
# Only run one capacity point: 0.5 / 0.5
capacities = np.array([0.5])

# number of total repeated runs
n_runs_per_point = 10

# save rolling average error every 2 runs
save_every = 2

eta_val = 0.5
output_dir = f"more_scale_batch_queue_results"
os.makedirs(output_dir, exist_ok=True)

print(f"batch_size = {batch_size}")
print(f"capacities = {capacities}")
print(f"policies = ['Logistic greedy', 'Tree greedy']")

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
print("Test size:", len(Xc_test))
# ============================================================
# 3. TRAIN TRUE EXPERT MODELS
# ============================================================
print("\n--- Step 3: Training expert models ---")

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

# ============================================================
# 4. TEST ENVIRONMENT
# ============================================================
rewards_matrix_test = np.column_stack([
    (preds_1_test == y_test).astype(int),
    (preds_2_test == y_test).astype(int)
])

# ============================================================
# 5. ONLINE LEARNER CLASSES
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
# 6. MCMF ASSIGNMENT
# ============================================================
def assign_mcmf_scores(scores, caps):
    n_agents, n_tasks = scores.shape
    if sum(caps) != n_tasks:
        raise ValueError("For exact assignment, sum(caps) must equal number of tasks in the batch.")

    src, sink = 0, 1
    task_off = 2
    agent_off = 2 + n_tasks

    smcf = min_cost_flow.SimpleMinCostFlow()

    for j in range(n_tasks):
        smcf.add_arc_with_capacity_and_unit_cost(src, task_off + j, 1, 0)

    scale = 10**6
    for i in range(n_agents):
        for j in range(n_tasks):
            cost = -int(np.round(scores[i, j] * scale))
            smcf.add_arc_with_capacity_and_unit_cost(task_off + j, agent_off + i, 1, cost)

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
# 7. BATCH CAPACITY RULE
# ============================================================
def batch_caps_from_queue(batch_size, p_targets, Q):
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
# ============================================================
def run_batch_queue_simulation(
    X,
    rewards_matrix,
    p_targets,
    perm,
    batch_size=100,
    n_runs=1,
    eta=1.0,
    mode='log_greedy',
    seed=42
):
    n_steps, dim = X.shape
    n_agents = len(p_targets)
    rng_local = np.random.default_rng(seed)

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

        if mode == 'tree_greedy':
            agents = [AgentPosteriorRewardTreeTS(rng=rng_local) for _ in range(n_agents)]
        elif mode == 'log_greedy':
            agents = [ContextualLogRegAgent(dim, alpha=0.5) for _ in range(n_agents)]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        cumulative_reward = 0.0
        step_rewards = np.zeros(n_steps)
        step_actions = np.zeros(n_steps, dtype=int)
        batch_queues = []
        batch_counts = []

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_steps)
            current_batch_size = end - start

            X_batch = X_shuffled[start:end]
            R_batch = rewards_shuffled[start:end]

            pred_scores = np.zeros((n_agents, current_batch_size))

            for j in range(current_batch_size):
                x_j = X_batch[j]
                for i in range(n_agents):
                    if mode == 'log_greedy':
                        pred_scores[i, j] = agents[i].predict_reward_prob(x_j)
                    elif mode == 'tree_greedy':
                        pred_scores[i, j] = agents[i].predict_reward(x_j)

            penalized_scores = pred_scores - eta * Q.reshape(-1, 1)
            caps_batch = batch_caps_from_queue(current_batch_size, p_targets, Q)

            if caps_batch.sum() != current_batch_size:
                raise RuntimeError("caps_batch does not sum to current_batch_size")

            assignment, _ = assign_mcmf_scores(penalized_scores, caps_batch)

            n_assigned = np.zeros(n_agents, dtype=int)

            for j in range(current_batch_size):
                global_j = start + j
                a = assignment[j]
                reward = R_batch[j, a]

                step_actions[global_j] = a
                step_rewards[global_j] = reward
                cumulative_reward += reward
                n_assigned[a] += 1

                agents[a].update(X_batch[j], reward)

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
# 9. EXPERIMENT LOOP
# ============================================================
online_stats = {
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy': {cap: [] for cap in capacities}
}

queue_logs = {
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy': {cap: [] for cap in capacities}
}

count_logs = {
    'Logistic greedy': {cap: [] for cap in capacities},
    'Tree greedy': {cap: [] for cap in capacities}
}

print("Running experiments...")

for r in range(1, n_runs_per_point + 1):
    print(f"\n=== Run {r}/{n_runs_per_point} ===")
    rng_run = np.random.default_rng(seed + r)
    perm_r = rng_run.permutation(len(Xc_test))

    for cap in capacities:
        targets = np.array([cap, 1.0 - cap])

        # -----------------------------------
        # Logistic greedy
        # -----------------------------------
        avg, _, _, q_hist, c_hist = run_batch_queue_simulation(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, n_runs=1, eta=eta_val,
            mode='log_greedy', seed=seed + 2000 + r
        )
        online_stats['Logistic greedy'][cap].append(avg)
        queue_logs['Logistic greedy'][cap].append(q_hist)
        count_logs['Logistic greedy'][cap].append(c_hist)

        # -----------------------------------
        # Tree greedy
        # -----------------------------------
        avg, _, _, q_hist, c_hist = run_batch_queue_simulation(
            Xc_test, rewards_matrix_test, targets, perm_r,
            batch_size=batch_size, n_runs=1, eta=eta_val,
            mode='tree_greedy', seed=seed + 4000 + r
        )
        online_stats['Tree greedy'][cap].append(avg)
        queue_logs['Tree greedy'][cap].append(q_hist)
        count_logs['Tree greedy'][cap].append(c_hist)

    # ===== MODIFIED =====
    # Save average error rate every 2 runs
    if r % save_every == 0:
        checkpoint_results = {
            'runs_completed': r,
            'capacity_agent_1': 0.5,
            'capacity_agent_2': 0.5,
            'Logistic greedy': {
                'avg_error': 1.0 - float(np.mean(online_stats['Logistic greedy'][0.5]))
            },
            'Tree greedy': {
                'avg_error': 1.0 - float(np.mean(online_stats['Tree greedy'][0.5]))
            }
        }

        checkpoint_path = os.path.join(
            output_dir,
            f"batch_queue_seed_{seed}_batch_{batch_size}_avg_error_through_run_{r}.json"
        )
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_results, f, indent=2)

        print(f"Saved checkpoint: {checkpoint_path}")
        print(json.dumps(checkpoint_results, indent=2))

# ============================================================
# 10. FINAL RESULTS
# ============================================================
results_final = {
    'runs_completed': n_runs_per_point,
    'capacity_agent_1': 0.5,
    'capacity_agent_2': 0.5,
    'Logistic greedy': {
        'avg_error': 1.0 - float(np.mean(online_stats['Logistic greedy'][0.5]))
    },
    'Tree greedy': {
        'avg_error': 1.0 - float(np.mean(online_stats['Tree greedy'][0.5]))
    }
}

final_json_path = os.path.join(
    output_dir,
    f"batch_queue_seed_{seed}_batch_{batch_size}_final.json"
)
with open(final_json_path, "w") as f:
    json.dump(results_final, f, indent=2)

np.save(
    os.path.join(output_dir, f"batch_queue_seed_{seed}_batch_{batch_size}_queue_logs.npy"),
    queue_logs,
    allow_pickle=True
)
np.save(
    os.path.join(output_dir, f"batch_queue_seed_{seed}_batch_{batch_size}_count_logs.npy"),
    count_logs,
    allow_pickle=True
)

print("\nDone.")
print("Saved results to:", output_dir)
print(json.dumps(results_final, indent=2))
