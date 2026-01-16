from __future__ import annotations

import numpy as np

from Day_1.linucb import LinUCBAgent
from Day_3.linear_thompson import LinearThompson


def block_features(X: np.ndarray) -> np.ndarray:
    """Turn (K, d) contexts into (K, K*d) block features."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    K, d = X.shape
    Xb = np.zeros((K, K * d), dtype=np.float64)
    # Each arm gets its own block so a single linear model learns per-arm weights.
    for a in range(K):
        Xb[a, a * d : (a + 1) * d] = X[a]
    return Xb


class EpsilonGreedyLinear:
    """Epsilon-greedy linear model using ridge regression updates."""

    def __init__(self, d: int, epsilon: float = 0.1, lam: float = 1.0):
        """Initialize epsilon-greedy linear model."""
        self.d = int(d)
        self.epsilon = float(epsilon)
        self.lam = float(lam)
        self.A_inv = (1.0 / self.lam) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)

    def theta_hat(self) -> np.ndarray:
        """Return current ridge-regression weights."""
        return self.A_inv @ self.b

    def greedy_arm(self, X: np.ndarray) -> int:
        """Pick the best arm under the current linear model."""
        th = self.theta_hat()
        scores = X @ th
        return int(np.argmax(scores))

    def select_arm(self, X: np.ndarray, rng: np.random.Generator) -> int:
        """Pick a random arm with epsilon, else greedy."""
        if rng.random() < self.epsilon:
            return int(rng.integers(0, X.shape[0]))
        return self.greedy_arm(X)

    def update(self, x: np.ndarray, r: float) -> None:
        """Update the ridge-regression inverse with one sample."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        # Sherman-Morrison update for the ridge-regression inverse.
        z = self.A_inv @ x
        self.A_inv -= np.outer(z, z) / (1.0 + float(x @ z))
        self.b += float(r) * x


class LinUCBWrapper:
    """Wrap LinUCB to provide a consistent interface."""

    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0):
        """Initialize LinUCB with the shared interface."""
        self.agent = LinUCBAgent(d=d, alpha=alpha, lam=lam)

    def select_arm(self, X: np.ndarray) -> int:
        """Pick an arm using UCB scores."""
        return self.agent.select_arm(X)

    def greedy_arm(self, X: np.ndarray) -> int:
        """Reuse UCB selection as the greedy choice."""
        return self.agent.select_arm(X)

    def update(self, x: np.ndarray, r: float) -> None:
        """Update LinUCB with an observed reward."""
        self.agent.update(x, r)

    @property
    def A_inv(self) -> np.ndarray:
        """Expose the inverse covariance for diagnostics."""
        return self.agent.A_inv

    @property
    def b(self) -> np.ndarray:
        """Expose the linear term for diagnostics."""
        return self.agent.b

    @property
    def d(self) -> int:
        """Expose the feature dimension."""
        return self.agent.d


def select_arm(agent, X: np.ndarray, rng: np.random.Generator) -> int:
    """Dispatch select_arm for the agent type."""
    if isinstance(agent, LinearThompson):
        return agent.select_arm(X, rng)
    if isinstance(agent, EpsilonGreedyLinear):
        return agent.select_arm(X, rng)
    return agent.select_arm(X)


def greedy_arm(agent, X: np.ndarray) -> int:
    """Dispatch greedy_arm for the agent type."""
    if isinstance(agent, LinearThompson):
        return agent.greedy_arm(X)
    if isinstance(agent, EpsilonGreedyLinear):
        return agent.greedy_arm(X)
    return agent.greedy_arm(X)
