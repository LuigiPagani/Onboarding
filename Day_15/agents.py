from __future__ import annotations

import numpy as np

from Day_1.linucb import LinUCBAgent


def block_features(X: np.ndarray) -> np.ndarray:
    """Turn (K, d) contexts into (K, K*d) block features."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    K, d = X.shape
    Xb = np.zeros((K, K * d), dtype=np.float64)
    for a in range(K):
        Xb[a, a * d : (a + 1) * d] = X[a]
    return Xb


class LinearThompson:
    """Linear Thompson Sampling with Gaussian prior/posterior."""

    def __init__(self, d: int, lam: float = 1.0, sigma: float = 0.5):
        self.d = int(d)
        self.lam = float(lam)
        self.sigma = float(sigma)
        self.A_inv = (1.0 / self.lam) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        mean = self.A_inv @ self.b
        cov = self.A_inv
        return rng.multivariate_normal(mean, cov)

    def select_arm(self, X: np.ndarray, rng: np.random.Generator) -> int:
        theta = self.sample_theta(rng)
        scores = X @ theta
        return int(np.argmax(scores))

    def greedy_arm(self, X: np.ndarray) -> int:
        mean = self.A_inv @ self.b
        scores = X @ mean
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, r: float) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        scale = 1.0 / (self.sigma ** 2)
        z = self.A_inv @ x
        denom = 1.0 + scale * float(x @ z)
        self.A_inv -= (scale * np.outer(z, z)) / denom
        self.b += scale * float(r) * x


class EpsilonGreedyLinear:
    """Epsilon-greedy linear model using ridge regression updates."""

    def __init__(self, d: int, epsilon: float = 0.1, lam: float = 1.0):
        self.d = int(d)
        self.epsilon = float(epsilon)
        self.lam = float(lam)
        self.A_inv = (1.0 / self.lam) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)

    def theta_hat(self) -> np.ndarray:
        return self.A_inv @ self.b

    def greedy_arm(self, X: np.ndarray) -> int:
        th = self.theta_hat()
        scores = X @ th
        return int(np.argmax(scores))

    def select_arm(self, X: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(0, X.shape[0]))
        return self.greedy_arm(X)

    def update(self, x: np.ndarray, r: float) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        z = self.A_inv @ x
        self.A_inv -= np.outer(z, z) / (1.0 + float(x @ z))
        self.b += float(r) * x


class LinUCBWrapper:
    """Wrap LinUCB to provide a consistent interface."""

    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0):
        self.agent = LinUCBAgent(d=d, alpha=alpha, lam=lam)

    def select_arm(self, X: np.ndarray) -> int:
        return self.agent.select_arm(X)

    def greedy_arm(self, X: np.ndarray) -> int:
        return self.agent.select_arm(X)

    def update(self, x: np.ndarray, r: float) -> None:
        self.agent.update(x, r)

    @property
    def A_inv(self) -> np.ndarray:
        return self.agent.A_inv

    @property
    def b(self) -> np.ndarray:
        return self.agent.b

    @property
    def d(self) -> int:
        return self.agent.d


def select_arm(agent, X: np.ndarray, rng: np.random.Generator) -> int:
    if isinstance(agent, LinearThompson):
        return agent.select_arm(X, rng)
    if isinstance(agent, EpsilonGreedyLinear):
        return agent.select_arm(X, rng)
    return agent.select_arm(X)


def greedy_arm(agent, X: np.ndarray) -> int:
    if isinstance(agent, LinearThompson):
        return agent.greedy_arm(X)
    if isinstance(agent, EpsilonGreedyLinear):
        return agent.greedy_arm(X)
    return agent.greedy_arm(X)
