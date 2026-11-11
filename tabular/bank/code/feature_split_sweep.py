import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# ============================================================
# 1. COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one banking feature-split experiment. "
            "Each process handles one feature fraction and one split seed."
        )
    )
    parser.add_argument(
        "feature_fraction",
        type=float,
        help="Requested fraction of raw features assigned to Agent 1.",
    )
    parser.add_argument(
        "split_seed",
        type=int,
        help="Seed used to randomly assign features to the two agents.",
    )
    parser.add_argument(
        "n_runs",
        type=int,
        nargs="?",
        default=10,
        help="Number of online stream permutations. Default: 10.",
    )
    parser.add_argument(
        "--data-file",
        default="../data/bank_marketing_raw.csv",
        help="Banking data file. Default: ../data/bank_marketing_raw.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="feature_split_results",
        help="Output directory. Default: feature_split_results.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.5,
        help="Virtual-queue penalty. Default: 0.5.",
    )
    return parser.parse_args()


args = parse_args()

feature_fraction = args.feature_fraction
split_seed = args.split_seed
n_runs = args.n_runs
data_file = args.data_file
output_dir = Path(args.output_dir)
eta_val = args.eta

if not 0.0 < feature_fraction < 1.0:
    raise ValueError("feature_fraction must be strictly between 0 and 1.")
if split_seed < 0:
    raise ValueError("split_seed must be nonnegative.")
if n_runs < 1:
    raise ValueError("n_runs must be at least 1.")

# Fixed capacity requested by the user.
capacity_agent1 = 0.5
capacity_agent2 = 0.5
capacity_targets = np.array([capacity_agent1, capacity_agent2])

# Reproducibility.
np.random.seed(split_seed)

output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FEATURE DEFINITIONS
# ============================================================

categorical_features = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
    "day_of_week",
]

continuous_features = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
]


def split_features_randomly(
    cat_features,
    cont_features,
    rng,
    frac,
):
    """
    Randomly assign approximately `frac` of the 15 raw features to Agent 1.
    Agent 2 receives all remaining features.

    Since there are 15 raw features, some requested fractions cannot be exact.
    The code uses the closest valid nonempty split and records the realized share.
    """
    all_features = cat_features + cont_features
    n_total = len(all_features)

    # Nearest valid integer number of features, with .5 rounded upward.
    n_agent1 = int(np.floor(frac * n_total + 0.5))
    n_agent1 = max(1, min(n_total - 1, n_agent1))

    agent1_features = rng.choice(
        all_features,
        size=n_agent1,
        replace=False,
    ).tolist()

    agent2_features = [
        feature
        for feature in all_features
        if feature not in agent1_features
    ]

    agent1_cat = [
        feature
        for feature in agent1_features
        if feature in cat_features
    ]
    agent1_cont = [
        feature
        for feature in agent1_features
        if feature in cont_features
    ]
    agent2_cat = [
        feature
        for feature in agent2_features
        if feature in cat_features
    ]
    agent2_cont = [
        feature
        for feature in agent2_features
        if feature in cont_features
    ]

    realized_fraction = n_agent1 / n_total

    return (
        agent1_cat,
        agent1_cont,
        agent2_cat,
        agent2_cont,
        realized_fraction,
    )


def make_one_hot_encoder():
    """Compatibility with both older and newer scikit-learn versions."""
    try:
        return OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
    except TypeError:
        return OneHotEncoder(
            sparse=False,
            handle_unknown="ignore",
        )


def encode_features(df, cat_features, cont_features):
    n = len(df)

    if cat_features:
        x_cat = make_one_hot_encoder().fit_transform(df[cat_features])
    else:
        x_cat = np.empty((n, 0))

    if cont_features:
        x_cont = StandardScaler().fit_transform(df[cont_features])
    else:
        x_cont = np.empty((n, 0))

    return np.hstack([x_cat, x_cont])


def load_and_prep_data(
    data_path,
    agent1_cat,
    agent1_cont,
    agent2_cat,
    agent2_cont,
):
    df = pd.read_csv(data_path, sep=";")
    df["target"] = (df["y"] == "yes").astype(int)

    # Each expert receives only its assigned features.
    x_agent1 = encode_features(
        df,
        agent1_cat,
        agent1_cont,
    )
    x_agent2 = encode_features(
        df,
        agent2_cat,
        agent2_cont,
    )

    # The contextual policy observes the full feature set.
    # Keeping this order fixed ensures the router context is identical
    # across all feature-split combinations.
    x_context = encode_features(
        df,
        categorical_features,
        continuous_features,
    )

    return train_test_split(
        x_context,
        x_agent1,
        x_agent2,
        df["target"].to_numpy(),
        test_size=0.3,
        random_state=42,
    )


# ============================================================
# 3. ONLINE POLICY CLASSES
# ============================================================

class ContextualLogRegAgent:
    def __init__(
        self,
        dimension,
        alpha=1.0,
        lambda_prior=1.0,
        rng=None,
    ):
        self.dimension = dimension
        self.alpha = alpha
        self.mean = np.zeros(dimension)
        self.a_inv = np.eye(dimension) / lambda_prior
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_prob(self, x):
        covariance = (self.alpha ** 2) * self.a_inv
        covariance = (covariance + covariance.T) / 2.0

        try:
            theta = self.rng.multivariate_normal(
                self.mean,
                covariance,
            )
        except np.linalg.LinAlgError:
            jitter = 1e-6 * np.eye(self.dimension)
            theta = self.rng.multivariate_normal(
                self.mean,
                covariance + jitter,
            )

        return expit(np.dot(x, theta))

    def predict_reward_prob(self, x):
        return expit(np.dot(self.mean, x))

    def update(self, x, reward):
        p_hat = expit(np.dot(self.mean, x))

        weight = p_hat * (1.0 - p_hat)
        weight = np.clip(weight, 1e-4, 0.25)

        a_inv_x = self.a_inv @ x
        numerator = weight * np.outer(a_inv_x, a_inv_x)
        denominator = 1.0 + weight * np.dot(x, a_inv_x)

        self.a_inv = self.a_inv - numerator / denominator
        self.a_inv = (self.a_inv + self.a_inv.T) / 2.0

        gradient = (reward - p_hat) * x
        self.mean = self.mean + self.a_inv @ gradient


class AgentPosteriorRewardTreeTS:
    def __init__(
        self,
        n_trees=20,
        max_depth=3,
        min_samples_leaf=10,
        refit_every=20,
        rng=None,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.refit_every = refit_every
        self.rng = rng if rng is not None else np.random.default_rng()

        self.x_history = []
        self.y_history = []
        self.t = 0
        self.fitted = False

        self.trees = [
            DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=i,
            )
            for i in range(n_trees)
        ]

    def sample_reward(self, x):
        if not self.fitted:
            return self.rng.random()

        tree_index = self.rng.integers(self.n_trees)
        prediction = self.trees[tree_index].predict(
            x.reshape(1, -1)
        )[0]

        return float(np.clip(prediction, 0.0, 1.0))

    def predict_reward(self, x):
        if not self.fitted:
            return 0.5

        predictions = [
            tree.predict(x.reshape(1, -1))[0]
            for tree in self.trees
        ]

        return float(
            np.clip(np.mean(predictions), 0.0, 1.0)
        )

    def update(self, x, reward):
        self.x_history.append(x)
        self.y_history.append(reward)
        self.t += 1

        if self.t % self.refit_every != 0:
            return

        x_array = np.asarray(self.x_history)
        y_array = np.asarray(self.y_history)
        n = len(y_array)

        for tree in self.trees:
            bootstrap_indices = self.rng.integers(
                0,
                n,
                size=n,
            )
            tree.fit(
                x_array[bootstrap_indices],
                y_array[bootstrap_indices],
            )

        self.fitted = True


# ============================================================
# 4. ONLINE SIMULATION
# ============================================================

def run_simulation(
    x_context,
    rewards_matrix,
    p_targets,
    permutation,
    eta,
    mode,
    policy_seed,
):
    """
    Run one online policy on one fixed stream permutation.

    Supported policies only:
      - log_ts
      - log_greedy
      - tree_ts
      - tree_greedy

    No random policy is simulated.
    """
    supported_modes = {
        "log_ts",
        "log_greedy",
        "tree_ts",
        "tree_greedy",
    }
    if mode not in supported_modes:
        raise ValueError(
            f"Unsupported mode {mode}. "
            f"Choose from {sorted(supported_modes)}."
        )

    n_steps, dimension = x_context.shape
    n_agents = len(p_targets)

    x_shuffled = x_context[permutation]
    rewards_shuffled = rewards_matrix[permutation]

    queue = np.zeros(n_agents)
    cumulative_reward = 0.0

    if mode in {"tree_ts", "tree_greedy"}:
        agents = [
            AgentPosteriorRewardTreeTS(
                rng=np.random.default_rng(
                    policy_seed + 1000 * (agent_index + 1)
                )
            )
            for agent_index in range(n_agents)
        ]
    else:
        agents = [
            ContextualLogRegAgent(
                dimension,
                alpha=0.5,
                rng=np.random.default_rng(
                    policy_seed + 1000 * (agent_index + 1)
                ),
            )
            for agent_index in range(n_agents)
        ]

    for t in range(n_steps):
        x_t = x_shuffled[t]
        scores = np.zeros(n_agents)

        if mode == "tree_ts":
            for i in range(n_agents):
                scores[i] = (
                    agents[i].sample_reward(x_t)
                    - eta * queue[i]
                )

        elif mode == "tree_greedy":
            for i in range(n_agents):
                scores[i] = (
                    agents[i].predict_reward(x_t)
                    - eta * queue[i]
                )

        elif mode == "log_ts":
            for i in range(n_agents):
                scores[i] = (
                    agents[i].sample_prob(x_t)
                    - eta * queue[i]
                )

        elif mode == "log_greedy":
            for i in range(n_agents):
                scores[i] = (
                    agents[i].predict_reward_prob(x_t)
                    - eta * queue[i]
                )

        action = int(np.argmax(scores))
        reward = rewards_shuffled[t, action]
        cumulative_reward += reward

        agents[action].update(x_t, reward)

        assignment = np.zeros(n_agents)
        assignment[action] = 1.0
        queue = np.maximum(
            queue + assignment - p_targets,
            0.0,
        )

    return cumulative_reward / n_steps


# ============================================================
# 5. CREATE THE FEATURE SPLIT
# ============================================================

split_rng = np.random.default_rng(split_seed)

(
    agent1_cat,
    agent1_cont,
    agent2_cat,
    agent2_cont,
    realized_fraction_agent1,
) = split_features_randomly(
    categorical_features,
    continuous_features,
    split_rng,
    feature_fraction,
)

agent1_features = agent1_cat + agent1_cont
agent2_features = agent2_cat + agent2_cont
n_total_features = (
    len(categorical_features)
    + len(continuous_features)
)

print("\n============================================================")
print("BANKING FEATURE-SPLIT EXPERIMENT")
print("============================================================")
print(f"Requested Agent 1 feature fraction: {feature_fraction:.3f}")
print(
    "Realized Agent 1 feature fraction: "
    f"{realized_fraction_agent1:.3f} "
    f"({len(agent1_features)}/{n_total_features} raw features)"
)
print(
    "Realized Agent 2 feature fraction: "
    f"{1.0 - realized_fraction_agent1:.3f} "
    f"({len(agent2_features)}/{n_total_features} raw features)"
)
print(f"Feature-split seed: {split_seed}")
print(f"Capacity targets: {capacity_targets.tolist()}")
print(f"Number of online runs: {n_runs}")
print(f"Queue eta: {eta_val}")

print("\nAgent 1 features")
print("  Categorical:", agent1_cat)
print("  Continuous :", agent1_cont)

print("\nAgent 2 features")
print("  Categorical:", agent2_cat)
print("  Continuous :", agent2_cont)
print("============================================================\n")


# ============================================================
# 6. LOAD DATA AND TRAIN THE TWO EXPERT AGENTS
# ============================================================

(
    x_context_train,
    x_context_test,
    x_agent1_train,
    x_agent1_test,
    x_agent2_train,
    x_agent2_test,
    y_train,
    y_test,
) = load_and_prep_data(
    data_file,
    agent1_cat,
    agent1_cont,
    agent2_cat,
    agent2_cont,
)

print("Training expert agents...")

# Agent models: code agent 1 = logistic regression (paper Agent 2),
#               code agent 2 = XGBoost (paper Agent 1).
model_agent1 = LogisticRegression(
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
)
model_agent2 = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=split_seed,
    n_jobs=1,
)

model_agent1.fit(x_agent1_train, y_train)
model_agent2.fit(x_agent2_train, y_train)

predictions_agent1 = model_agent1.predict(x_agent1_test)
predictions_agent2 = model_agent2.predict(x_agent2_test)

reward_agent1 = (
    predictions_agent1 == y_test
).astype(int)
reward_agent2 = (
    predictions_agent2 == y_test
).astype(int)

rewards_matrix_test = np.column_stack([
    reward_agent1,
    reward_agent2,
])


# ============================================================
# 7. REALIZED AGENT HETEROGENEITY
# ============================================================

agent1_accuracy = float(reward_agent1.mean())
agent2_accuracy = float(reward_agent2.mean())

# p1: Agent 1 correct and Agent 2 wrong.
p1 = float(
    np.mean(reward_agent1 > reward_agent2)
)

# p2: Agent 2 correct and Agent 1 wrong.
p2 = float(
    np.mean(reward_agent2 > reward_agent1)
)

disagreement_rate = float(
    np.mean(reward_agent1 != reward_agent2)
)

both_correct_rate = float(
    np.mean(
        (reward_agent1 == 1)
        & (reward_agent2 == 1)
    )
)

both_wrong_rate = float(
    np.mean(
        (reward_agent1 == 0)
        & (reward_agent2 == 0)
    )
)

# No random policy is run.
# This is the expected error from assigning each observation to each
# agent with probability 0.5, computed directly from marginal accuracies.
random_error = float(
    capacity_agent1 * (1.0 - agent1_accuracy)
    + capacity_agent2 * (1.0 - agent2_accuracy)
)

print("\n============================================================")
print("REALIZED AGENT HETEROGENEITY")
print("============================================================")
print(f"Agent 1 marginal accuracy: {agent1_accuracy:.6f}")
print(f"Agent 2 marginal accuracy: {agent2_accuracy:.6f}")
print(f"p1 (Agent 1 correct, Agent 2 wrong): {p1:.6f}")
print(f"p2 (Agent 2 correct, Agent 1 wrong): {p2:.6f}")
print(f"Disagreement rate: {disagreement_rate:.6f}")
print(f"Check, p1 + p2: {p1 + p2:.6f}")
print(f"Both correct rate: {both_correct_rate:.6f}")
print(f"Both wrong rate: {both_wrong_rate:.6f}")
print(f"Analytical 50/50 random error: {random_error:.6f}")
print("============================================================\n")


# ============================================================
# 8. RUN ONLY THE FOUR REQUESTED ONLINE POLICIES
# ============================================================

policies = {
    "Logistic TS": "log_ts",
    "Logistic greedy": "log_greedy",
    "Tree TS": "tree_ts",
    "Tree greedy": "tree_greedy",
}

policy_seed_offsets = {
    "Logistic TS": 101,
    "Logistic greedy": 202,
    "Tree TS": 303,
    "Tree greedy": 404,
}

per_run_results = []

print("Running online policies...")

for run_index in range(n_runs):
    # The same stream permutation is used across all four policies.
    # It is also held constant across feature-split jobs for a given run.
    stream_seed = 100000 + run_index
    stream_rng = np.random.default_rng(stream_seed)
    permutation = stream_rng.permutation(
        len(x_context_test)
    )

    print(
        f"\n=== Run {run_index + 1}/{n_runs}; "
        f"stream seed={stream_seed} ==="
    )

    for policy_name, policy_mode in policies.items():
        policy_seed = (
            10000000
            + 10000 * split_seed
            + 100 * run_index
            + policy_seed_offsets[policy_name]
        )

        average_reward = run_simulation(
            x_context=x_context_test,
            rewards_matrix=rewards_matrix_test,
            p_targets=capacity_targets,
            permutation=permutation,
            eta=eta_val,
            mode=policy_mode,
            policy_seed=policy_seed,
        )

        error_rate = float(1.0 - average_reward)
        gain_over_random = float(
            random_error - error_rate
        )

        per_run_results.append({
            "requested_fraction_agent1": feature_fraction,
            "realized_fraction_agent1": realized_fraction_agent1,
            "split_seed": split_seed,
            "capacity_agent1": capacity_agent1,
            "capacity_agent2": capacity_agent2,
            "eta": eta_val,
            "run": run_index + 1,
            "stream_seed": stream_seed,
            "policy": policy_name,
            "error_rate": error_rate,
            "random_error": random_error,
            "gain_over_random": gain_over_random,
        })

        print(
            f"{policy_name}: "
            f"error={error_rate:.6f}, "
            f"gain over random={gain_over_random:.6f}"
        )


# ============================================================
# 9. CREATE SUMMARY TABLE
# ============================================================

per_run_df = pd.DataFrame(per_run_results)

summary_results = []

for policy_name in policies:
    policy_rows = per_run_df[
        per_run_df["policy"] == policy_name
    ]

    errors = policy_rows["error_rate"].to_numpy()
    gains = policy_rows["gain_over_random"].to_numpy()

    if len(errors) > 1:
        error_se = float(
            np.std(errors, ddof=1)
            / np.sqrt(len(errors))
        )
        gain_se = float(
            np.std(gains, ddof=1)
            / np.sqrt(len(gains))
        )
    else:
        error_se = np.nan
        gain_se = np.nan

    summary_results.append({
        "requested_fraction_agent1": feature_fraction,
        "realized_fraction_agent1": realized_fraction_agent1,
        "realized_fraction_agent2": 1.0 - realized_fraction_agent1,
        "n_features_agent1": len(agent1_features),
        "n_features_agent2": len(agent2_features),
        "split_seed": split_seed,
        "n_runs": n_runs,
        "capacity_agent1": capacity_agent1,
        "capacity_agent2": capacity_agent2,
        "eta": eta_val,
        "agent1_features": json.dumps(agent1_features),
        "agent2_features": json.dumps(agent2_features),
        "agent1_categorical_features": json.dumps(agent1_cat),
        "agent1_continuous_features": json.dumps(agent1_cont),
        "agent2_categorical_features": json.dumps(agent2_cat),
        "agent2_continuous_features": json.dumps(agent2_cont),
        "agent1_accuracy": agent1_accuracy,
        "agent2_accuracy": agent2_accuracy,
        "p1": p1,
        "p2": p2,
        "disagreement_rate": disagreement_rate,
        "both_correct_rate": both_correct_rate,
        "both_wrong_rate": both_wrong_rate,
        "random_error": random_error,
        "policy": policy_name,
        "mean_error": float(np.mean(errors)),
        "se_error": error_se,
        "mean_gain_over_random": float(np.mean(gains)),
        "se_gain_over_random": gain_se,
    })

summary_df = pd.DataFrame(summary_results)


# ============================================================
# 10. SAVE THIS JOB'S OUTPUTS
# ============================================================

fraction_label = (
    f"{feature_fraction:.3f}"
    .rstrip("0")
    .rstrip(".")
    .replace(".", "p")
)

job_label = (
    f"banking_frac_{fraction_label}"
    f"_seed_{split_seed}"
)

per_run_file = (
    output_dir
    / f"{job_label}_per_run.csv"
)

summary_file = (
    output_dir
    / f"{job_label}_summary.csv"
)

per_run_df.to_csv(
    per_run_file,
    index=False,
)

summary_df.to_csv(
    summary_file,
    index=False,
)

print("\n============================================================")
print("FINAL POLICY SUMMARY")
print("============================================================")
print(
    summary_df[
        [
            "policy",
            "mean_error",
            "se_error",
            "mean_gain_over_random",
            "se_gain_over_random",
        ]
    ].to_string(index=False)
)
print("============================================================")
print(f"Saved per-run results: {per_run_file}")
print(f"Saved summary results: {summary_file}")
