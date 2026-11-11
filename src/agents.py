import numpy as np
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor


class ContextualLogRegAgent:
    """Bayesian online logistic regression with Laplace approximation (Thompson sampling)."""

    def __init__(
        self,
        d: int,
        alpha: float = 1.0,
        lambda_prior: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.d = d
        self.alpha = alpha
        self.m = np.zeros(d)
        self.A_inv = np.eye(d) / lambda_prior
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_prob(self, x: np.ndarray) -> float:
        cov = (self.alpha ** 2) * self.A_inv
        cov = (cov + cov.T) / 2.0
        try:
            theta = self.rng.multivariate_normal(self.m, cov)
        except np.linalg.LinAlgError:
            theta = self.rng.multivariate_normal(self.m, cov + 1e-6 * np.eye(self.d))
        return float(expit(x @ theta))

    def predict_reward_prob(self, x: np.ndarray) -> float:
        return float(expit(self.m @ x))

    def update(self, x: np.ndarray, reward: float) -> None:
        p_hat = expit(self.m @ x)
        w = np.clip(p_hat * (1 - p_hat), 1e-4, 0.25)
        Ax = self.A_inv @ x
        self.A_inv -= (w * np.outer(Ax, Ax)) / (1 + w * (x @ Ax))
        self.A_inv = (self.A_inv + self.A_inv.T) / 2.0
        self.m = self.m + self.A_inv @ ((reward - p_hat) * x)


class AgentPosteriorRewardTreeTS:
    """Ensemble of decision trees with Thompson sampling via random tree selection."""

    def __init__(
        self,
        n_trees: int = 20,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        refit_every: int = 20,
        rng: np.random.Generator | None = None,
    ):
        self.n_trees = n_trees
        self.refit_every = refit_every
        self.rng = rng if rng is not None else np.random.default_rng()
        self.fitted = False
        self.X: list = []
        self.y: list = []
        self.t = 0
        # Each tree's split tie-breaking is seeded from self.rng so that no
        # randomness leaks in from numpy's global state.
        self.trees = [
            DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=int(self.rng.integers(2**31 - 1)),
            )
            for _ in range(n_trees)
        ]

    def sample_reward(self, x: np.ndarray) -> float:
        if not self.fitted:
            return float(self.rng.random())
        idx = int(self.rng.integers(self.n_trees))
        return float(np.clip(self.trees[idx].predict(x.reshape(1, -1))[0], 0.0, 1.0))

    def predict_reward(self, x: np.ndarray) -> float:
        if not self.fitted:
            return 0.5
        preds = [t.predict(x.reshape(1, -1))[0] for t in self.trees]
        return float(np.clip(np.mean(preds), 0.0, 1.0))

    def update(self, x: np.ndarray, reward: float) -> None:
        self.X.append(x)
        self.y.append(reward)
        self.t += 1
        if self.t % self.refit_every != 0:
            return
        X_arr, y_arr = np.array(self.X), np.array(self.y)
        n = len(y_arr)
        for tree in self.trees:
            idx = self.rng.integers(0, n, size=n)
            tree.fit(X_arr[idx], y_arr[idx])
        self.fitted = True
