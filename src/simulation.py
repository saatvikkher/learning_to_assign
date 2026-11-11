"""
Core simulation engine for the contextual routing bandit.

Each function runs a single trajectory and returns the average reward
(accuracy). Converting to error rate (1 - accuracy) is done in the
experiment runner.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from src.agents import ContextualLogRegAgent, AgentPosteriorRewardTreeTS

ONLINE_MODES = {"log_ts", "log_greedy", "tree_ts", "tree_greedy", "random"}
OFFLINE_MODES = {
    "offline_logistic",
    "offline_tree",
    "offline_logistic_unconstrained",
    "offline_tree_unconstrained",
}


def run_online(
    X: np.ndarray,
    rewards_matrix: np.ndarray,
    p_targets: np.ndarray,
    perm: np.ndarray,
    mode: str,
    eta: float = 1.0,
    rng: np.random.Generator | None = None,
    return_actions: bool = False,
) -> float | tuple[float, np.ndarray]:
    """One trajectory of an online policy. Returns average reward
    (and the action sequence if return_actions)."""
    if rng is None:
        rng = np.random.default_rng()

    n_steps, dim = X.shape
    n_k = len(p_targets)
    Q = np.zeros(n_k)

    if mode in ("tree_ts", "tree_greedy"):
        agents = [AgentPosteriorRewardTreeTS(rng=rng) for _ in range(n_k)]
    else:
        agents = [ContextualLogRegAgent(dim, alpha=0.5, rng=rng) for _ in range(n_k)]

    X_s = X[perm]
    R_s = rewards_matrix[perm]
    cumulative = 0.0
    actions = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        x_t = X_s[t]
        scores = np.zeros(n_k)

        if mode == "tree_ts":
            for i in range(n_k):
                scores[i] = agents[i].sample_reward(x_t) - eta * Q[i]
        elif mode == "tree_greedy":
            for i in range(n_k):
                scores[i] = agents[i].predict_reward(x_t) - eta * Q[i]
        elif mode == "log_ts":
            for i in range(n_k):
                scores[i] = agents[i].sample_prob(x_t) - eta * Q[i]
        elif mode == "log_greedy":
            for i in range(n_k):
                scores[i] = agents[i].predict_reward_prob(x_t) - eta * Q[i]

        if mode == "random":
            action = int(rng.choice(n_k, p=p_targets))
        else:
            action = int(np.argmax(scores))

        reward = R_s[t, action]
        cumulative += reward
        actions[t] = action

        if mode != "random":
            agents[action].update(x_t, reward)

        z = np.zeros(n_k)
        z[action] = 1.0
        Q = np.maximum(Q + z - p_targets, 0.0)

    if return_actions:
        return cumulative / n_steps, actions
    return cumulative / n_steps


def run_offline(
    X: np.ndarray,
    rewards_matrix: np.ndarray,
    p_targets: np.ndarray,
    perm: np.ndarray,
    mode: str,
    offline_models: tuple,
    eta: float = 1.0,
) -> float:
    """One trajectory of an offline policy. Returns average reward."""
    off_log_1, off_log_2, off_tree_1, off_tree_2 = offline_models
    n_steps = len(X)
    n_k = len(p_targets)
    Q = np.zeros(n_k)
    X_s = X[perm]
    R_s = rewards_matrix[perm]
    cumulative = 0.0
    constrained = "unconstrained" not in mode

    for t in range(n_steps):
        x_t = X_s[t].reshape(1, -1)
        if "logistic" in mode:
            s0 = off_log_1.predict_proba(x_t)[0, 1]
            s1 = off_log_2.predict_proba(x_t)[0, 1]
        else:
            s0 = off_tree_1.predict(x_t)[0]
            s1 = off_tree_2.predict(x_t)[0]

        if constrained:
            s0 -= eta * Q[0]
            s1 -= eta * Q[1]

        action = 0 if s0 >= s1 else 1
        cumulative += R_s[t, action]

        if constrained:
            z = np.zeros(n_k)
            z[action] = 1.0
            Q = np.maximum(Q + z - p_targets, 0.0)

    return cumulative / n_steps


def train_offline_models(
    X: np.ndarray,
    reward_target_1: np.ndarray,
    reward_target_2: np.ndarray,
    seed: int = 42,
) -> tuple:
    """Train offline logistic and tree baselines. Returns (log1, log2, tree1, tree2)."""
    log1 = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, reward_target_1)
    log2 = LogisticRegression(solver="lbfgs", max_iter=1000).fit(X, reward_target_2)
    tree1 = RandomForestRegressor(
        n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=seed
    ).fit(X, reward_target_1)
    tree2 = RandomForestRegressor(
        n_estimators=20, max_depth=3, min_samples_leaf=10, random_state=seed
    ).fit(X, reward_target_2)
    return log1, log2, tree1, tree2
