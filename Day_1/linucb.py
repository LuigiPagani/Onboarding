from __future__ import annotations

import numpy as np


class LinUCBAgent:
    """
    Simple LinUCB agent with Sherman–Morrison updates.

    Maintains:
      A_inv = (lam I + sum x x^T)^{-1}
      b     = sum x r

    Prediction:
      theta_hat = A_inv @ b

    Action:
      argmax_a x_a^T theta_hat + alpha * sqrt(x_a^T A_inv x_a)
    """

    def __init__(self, d: int, alpha: float = 1.0, lam: float = 1.0):
        """
        Initialize the LinUCB agent.

        Args:
            d: Dimension of the feature/context vectors.
            alpha: Exploration parameter controlling the width of the confidence bound.
            lam: Regularization parameter (lambda) for the ridge regression.
        """
        self.d = int(d)
        self.alpha = float(alpha)
        self.lam = float(lam)

        self.A_inv = (1.0 / self.lam) * np.eye(self.d, dtype=np.float64)
        self.b = np.zeros(self.d, dtype=np.float64)

    def theta_hat(self) -> np.ndarray:
        return self.A_inv @ self.b  # O(d^2)

    def select_arm(self, X: np.ndarray) -> int:
        """
        Select an arm.

        Args:
          X: (K, d) array of contexts/features for the K arms at current round.

        Returns:
          index (int) of chosen arm
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        if X.shape[1] != self.d:
            raise ValueError(f"X.shape[1]={X.shape[1]} must equal self.d={self.d}")

        th = self.theta_hat()
        scores = np.empty(X.shape[0], dtype=np.float64)

        for i, x in enumerate(X):
            mean = float(x @ th)
            Ainv_x = self.A_inv @ x
            bonus = self.alpha * float(np.sqrt(x @ Ainv_x))
            scores[i] = mean + bonus

        return int(np.argmax(scores))

    def update(self, x: np.ndarray, r: float) -> None:
        """
        Update the agent with chosen context x and observed reward r.

        Args:
          x: (d,) chosen arm context
          r: scalar reward
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.d:
            raise ValueError(f"x.shape[0]={x.shape[0]} must equal self.d={self.d}")

        # Sherman–Morrison: A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        z = self.A_inv @ x
        self.A_inv -= np.outer(z, z) / (1.0 + float(x @ z))  # O(d^2)

        # Update b
        self.b += float(r) * x


# Day 2 wording uses `LinearUCB`; keep it as an alias for the same implementation.
class LinearUCB(LinUCBAgent):
    pass


