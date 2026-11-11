"""
Trajectory-recording variants of the core simulation loops.

Same policies, same RNG consumption order, and same returned average reward as
``src.simulation`` — the only difference is that each step's chosen agent and
realized reward are also recorded and returned. Kept in a separate module so
the main experiment path in ``src.simulation`` is untouched.

Because the RNG draws happen in the same order as in ``run_online``, a
trajectory run seeded with ``[seed, run_idx, algo_idx + 1]`` reproduces exactly
the average reward that the corresponding main experiment logs for that
(capacity, run) pair.
"""

import numpy as np

from src.agents import ContextualLogRegAgent, AgentPosteriorRewardTreeTS


def run_online_traj(
    X: np.ndarray,
    rewards_matrix: np.ndarray,
    p_targets: np.ndarray,
    perm: np.ndarray,
    mode: str,
    eta: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    One trajectory of an online policy.

    Returns (average reward, step_actions, step_rewards) where step_actions[t]
    is the agent assigned at step t and step_rewards[t] the reward it realized.
    Both are indexed in permuted (arrival) order — ``perm`` maps them back to
    rows of ``X``.
    """
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

    step_actions = np.zeros(n_steps, dtype=np.uint8)
    step_rewards = np.zeros(n_steps, dtype=np.uint8)

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

        step_actions[t] = action
        step_rewards[t] = reward

        if mode != "random":
            agents[action].update(x_t, reward)

        z = np.zeros(n_k)
        z[action] = 1.0
        Q = np.maximum(Q + z - p_targets, 0.0)

    return cumulative / n_steps, step_actions, step_rewards


def run_offline_traj(
    X: np.ndarray,
    rewards_matrix: np.ndarray,
    p_targets: np.ndarray,
    perm: np.ndarray,
    mode: str,
    offline_models: tuple,
    eta: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """One trajectory of an offline policy. Returns (avg reward, actions, rewards)."""
    off_log_1, off_log_2, off_tree_1, off_tree_2 = offline_models
    n_steps = len(X)
    n_k = len(p_targets)
    Q = np.zeros(n_k)
    X_s = X[perm]
    R_s = rewards_matrix[perm]
    cumulative = 0.0
    constrained = "unconstrained" not in mode

    step_actions = np.zeros(n_steps, dtype=np.uint8)
    step_rewards = np.zeros(n_steps, dtype=np.uint8)

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
        reward = R_s[t, action]
        cumulative += reward

        step_actions[t] = action
        step_rewards[t] = reward

        if constrained:
            z = np.zeros(n_k)
            z[action] = 1.0
            Q = np.maximum(Q + z - p_targets, 0.0)

    return cumulative / n_steps, step_actions, step_rewards
