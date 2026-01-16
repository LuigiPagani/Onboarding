from __future__ import annotations

import numpy as np


class LinearThompson:
    """Linear Thompson Sampling with Gaussian prior/posterior."""

    def __init__(self, d: int, lam: float = 1.0, sigma: float = 0.5):
        """Initialize the Thompson Sampling posterior."""
        self.d = int(d)
        self.lam = float(lam)
        self.sigma = float(sigma)

        self.A_inv = (1.0 / self.lam) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)

    def sample_theta(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a parameter vector from the posterior."""
        mean = self.A_inv @ self.b
        cov = self.A_inv
        return rng.multivariate_normal(mean, cov)

    def select_arm(self, X: np.ndarray, rng: np.random.Generator) -> int:
        """Select the arm that maximizes the sampled linear score."""
        theta = self.sample_theta(rng)
        scores = X @ theta
        return int(np.argmax(scores))

    def greedy_arm(self, X: np.ndarray) -> int:
        """Select the arm that maximizes the posterior mean score."""
        mean = self.A_inv @ self.b
        scores = X @ mean
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, r: float) -> None:
        """Update the Gaussian posterior after observing reward r for features x."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        scale = 1.0 / (self.sigma ** 2)

        # Sherman-Morrison rank-1 update on the precision inverse.
        z = self.A_inv @ x
        denom = 1.0 + scale * float(x @ z)
        self.A_inv -= (scale * np.outer(z, z)) / denom
        self.b += scale * float(r) * x
