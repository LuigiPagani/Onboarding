from __future__ import annotations

import numpy as np
try:
    from sklearn.linear_model import LogisticRegression
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for DR OPE. Install with `pip install scikit-learn`."
    ) from exc


def _one_hot(actions: np.ndarray, n_actions: int) -> np.ndarray:
    """One-hot encode actions."""
    actions = np.asarray(actions, dtype=int).reshape(-1)
    out = np.zeros((actions.shape[0], n_actions), dtype=np.float64)
    out[np.arange(actions.shape[0]), actions] = 1.0
    return out


def build_action_features(
    contexts: np.ndarray, actions: np.ndarray, n_actions: int, use_interactions: bool = True
) -> np.ndarray:
    """Build features for the reward model.

    If use_interactions is True, build block features so each action has its own
    context weights. Otherwise, concatenate context features with action one-hot.
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    if not use_interactions:
        return np.concatenate([contexts, _one_hot(actions, n_actions)], axis=1)
    action_oh = _one_hot(actions, n_actions)
    # Outer product gives per-action context blocks; reshape to (n, n_actions * d).
    blocks = np.einsum("nk,nd->nkd", action_oh, contexts)
    return blocks.reshape(contexts.shape[0], n_actions * contexts.shape[1])


def fit_reward_model(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    n_actions: int,
    use_interactions: bool = True,
) -> LogisticRegression:
    """Fit a logistic reward model for DR OPE."""
    X = build_action_features(contexts, actions, n_actions, use_interactions)
    y = np.asarray(rewards, dtype=np.float64).reshape(-1)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def predict_all_actions(
    model: LogisticRegression,
    contexts: np.ndarray,
    n_actions: int,
    use_interactions: bool = True,
) -> np.ndarray:
    """Predict reward probabilities for every action in each context."""
    contexts = np.asarray(contexts, dtype=np.float64)
    preds = np.zeros((contexts.shape[0], n_actions), dtype=np.float64)
    for a in range(n_actions):
        X = build_action_features(
            contexts,
            np.full(contexts.shape[0], a),
            n_actions,
            use_interactions,
        )
        preds[:, a] = model.predict_proba(X)[:, 1]
    return preds


def dr_estimate(
    contexts: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
    q_hat: np.ndarray,
) -> float:
    """Compute a doubly-robust estimate for the target policy.

    DR = (1/n) * sum_i [
        q_hat(x_i, a*_i)
        + (a_i == a*_i) * (r_i - q_hat(x_i, a_i)) / p_i
    ]
    """
    contexts = np.asarray(contexts, dtype=np.float64)
    actions = np.asarray(actions, dtype=int).reshape(-1)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    prop = np.asarray(propensities, dtype=np.float64).reshape(-1)
    target_actions = np.asarray(target_actions, dtype=int).reshape(-1)

    idx = np.arange(contexts.shape[0])
    # Doubly Robust: direct model estimate + IPS correction when target matches logging action.
    direct = q_hat[idx, target_actions]
    correction = (target_actions == actions) * (rewards - q_hat[idx, actions]) / prop
    return float(np.mean(direct + correction))
