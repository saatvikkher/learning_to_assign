"""
Constrained-allocation baselines that use the same online reward models as our policies.

  run_online_plugin  Plug-in quantile threshold (Prop. 2.1): give agent 0 the tasks whose
                     estimated gap delta(x) = mu0(x) - mu1(x) is above the running
                     (1 - alpha_0) quantile of all gaps seen so far. Greedy point estimates,
                     updated from bandit feedback. No queue. Ported from
                     banking_plugin_10_runs.py (online part).
  run_dmd            Dual Mirror Descent for online allocation (Balseiro, Lu & Mirrokni 2023,
                     arXiv:2011.10124, Alg. 1) with hard budgets B_k = rho_k * T, Euclidean or
                     entropic mirror map, step 1/sqrt(T). Ported from banking_dmd_10_runs.py.

Each returns (avg_reward, actions). Randomness only flows through the given generators
(see CLAUDE.md RNG discipline).
"""

import numpy as np

from src.agents import ContextualLogRegAgent, AgentPosteriorRewardTreeTS


def _greedy_agents(kind: str, dim: int, rng: np.random.Generator):
    if kind == "tree":
        agents = [AgentPosteriorRewardTreeTS(rng=rng) for _ in range(2)]
        return agents, lambda a, x: a.predict_reward(x)
    agents = [ContextualLogRegAgent(dim, alpha=0.5, rng=rng) for _ in range(2)]
    return agents, lambda a, x: a.predict_reward_prob(x)


def threshold_rule(delta_ref: np.ndarray, cap0: float) -> tuple[float, float]:
    """(tau, q): agent 0 if delta > tau; if delta == tau, agent 0 w.p. q. Chosen so exactly a
    cap0 fraction of delta_ref goes to agent 0."""
    n = len(delta_ref)
    k = cap0 * n
    if k <= 0:
        return np.inf, 0.0
    if k >= n:
        return -np.inf, 1.0
    kc = int(np.ceil(k))
    tau = np.partition(delta_ref, n - kc)[n - kc]  # kc-th largest value
    n_above = np.count_nonzero(delta_ref > tau)
    n_equal = np.count_nonzero(delta_ref == tau)
    q = float(np.clip((k - n_above) / n_equal, 0.0, 1.0))
    return tau, q


def run_online_plugin(X, R, cap0, perm, kind, rng, rng_tie):
    Xs, Rs = X[perm], R[perm]
    n, d = Xs.shape
    agents, predict = _greedy_agents(kind, d, rng)
    hist = np.empty(n)
    actions = np.zeros(n, dtype=int)
    total = 0.0
    for t in range(n):
        x = Xs[t]
        delta = predict(agents[0], x) - predict(agents[1], x)
        hist[t] = delta
        tau, q = threshold_rule(hist[: t + 1], cap0)
        a = 0 if (delta > tau or (delta == tau and rng_tie.random() < q)) else 1
        r = Rs[t, a]
        agents[a].update(x, r)
        actions[t] = a
        total += r
    return total / n, actions


DMD_LAMBDA_INIT = {"euclidean": 0.0, "entropic": 0.1}  # entropic needs lambda > 0
DMD_LAMBDA_MIN = 1e-6


def run_dmd(X, R, cap0, perm, kind, mirror, rng, eta_dual=None, hard_budget=True):
    Xs, Rs = X[perm], R[perm]
    n, d = Xs.shape
    rho = np.array([cap0, 1.0 - cap0])
    budget = rho * n
    eta = 1.0 / np.sqrt(n) if eta_dual is None else eta_dual
    lam_max = 1.0 / rho.min()
    agents, predict = _greedy_agents(kind, d, rng)

    lam = np.full(2, DMD_LAMBDA_INIT[mirror])
    used = np.zeros(2)
    actions = np.zeros(n, dtype=int)
    n_forced = 0
    total = 0.0
    for t in range(n):
        x = Xs[t]
        scores = np.array([predict(agents[k], x) - lam[k] for k in range(2)])
        avail = (used < budget - 1e-9) if hard_budget else np.ones(2, dtype=bool)
        if not avail.any():  # numerical edge case at t = n-1
            avail[:] = True
        if not avail.all():
            n_forced += 1
            scores[~avail] = -np.inf
        a = int(np.argmax(scores))  # ties -> agent 0
        r = Rs[t, a]
        agents[a].update(x, r)
        actions[t] = a
        total += r
        used[a] += 1
        g = rho.copy()
        g[a] -= 1.0  # dual subgradient rho - e_a
        if mirror == "euclidean":
            lam = np.clip(lam - eta * g, 0.0, lam_max)
        else:
            lam = np.clip(lam * np.exp(-eta * g), DMD_LAMBDA_MIN, lam_max)
    return total / n, actions, {"n_forced": n_forced, "final_lambda": lam.tolist()}
