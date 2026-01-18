from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SimConfig:
    """Simulation configuration constants."""
    n_arms: int = 3
    embed_dim: int = 8
    category_count: int = 4
    persona_count: int = 3
    delay_mean: float = 50.0
    unsafe_rate: float = 0.02
    value_per_conversion: float = 1.0
    costs: Tuple[float, float, float] = (0.01, 0.02, 0.10)


CATEGORY_NAMES = ["electronics", "home", "fashion", "fitness"]
PERSONA_NAMES = ["budget", "balanced", "quality"]
MODEL_NAMES = ["A", "B", "C"]


def one_hot(idx: int, size: int) -> np.ndarray:
    """Return a one-hot vector of length size."""
    v = np.zeros(size, dtype=np.float64)
    v[int(idx)] = 1.0
    return v


def sample_context(rng: np.random.Generator, cfg: SimConfig) -> Dict[str, np.ndarray | int]:
    """Sample a user context with embedding, category, and persona."""
    embedding = rng.normal(size=cfg.embed_dim).astype(np.float64)
    category = int(rng.integers(0, cfg.category_count))
    persona = int(rng.integers(0, cfg.persona_count))
    return {"embedding": embedding, "category": category, "persona": persona}


def context_features(context: Dict[str, np.ndarray | int], cfg: SimConfig) -> np.ndarray:
    """Build the base feature vector for a context."""
    embed = np.asarray(context["embedding"], dtype=np.float64)
    category = int(context["category"])
    persona = int(context["persona"])
    feats = [
        embed,
        one_hot(category, cfg.category_count),
        one_hot(persona, cfg.persona_count),
        np.array([1.0], dtype=np.float64),
    ]
    return np.concatenate(feats)


def make_context_matrix(context: Dict[str, np.ndarray | int], cfg: SimConfig) -> np.ndarray:
    """Repeat the base features for each arm."""
    base = context_features(context, cfg)
    return np.repeat(base[None, :], cfg.n_arms, axis=0)


def sample_delay(rng: np.random.Generator, cfg: SimConfig) -> int:
    """Sample an integer delay for conversion feedback."""
    # Exponential delay to mimic long-tail feedback.
    if cfg.delay_mean <= 0:
        return 0
    delay = int(rng.exponential(cfg.delay_mean))
    return max(delay, 1)


def model_cost(action: int, cfg: SimConfig) -> float:
    """Return the cost for the chosen model."""
    return float(cfg.costs[int(action)])


def build_conversion_params(cfg: SimConfig, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Create hidden parameters for the conversion probability."""
    # Base weights for continuous embeddings per model.
    embed_w = rng.normal(scale=0.15, size=(cfg.n_arms, cfg.embed_dim)).astype(np.float64)

    # Persona affinity: budget favors A, quality favors C.
    persona_aff = np.array(
        [
            [0.6, 0.1, -0.4],
            [0.1, 0.4, 0.1],
            [-0.3, 0.2, 0.6],
        ],
        dtype=np.float64,
    )

    # Category affinity: simple pattern with slight preferences.
    category_aff = np.array(
        [
            [0.2, 0.0, -0.1],
            [0.0, 0.2, 0.1],
            [-0.1, 0.1, 0.3],
            [0.1, -0.1, 0.2],
        ],
        dtype=np.float64,
    )

    base = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
    return {"embed_w": embed_w, "persona_aff": persona_aff, "category_aff": category_aff, "base": base}


def sigmoid(x: float) -> float:
    """Sigmoid activation for probabilities."""
    return 1.0 / (1.0 + np.exp(-x))


def conversion_probability(
    context: Dict[str, np.ndarray | int],
    action: int,
    params: Dict[str, np.ndarray],
) -> float:
    """Compute conversion probability for a context/action pair."""
    embed = np.asarray(context["embedding"], dtype=np.float64)
    persona = int(context["persona"])
    category = int(context["category"])
    a = int(action)

    score = (
        params["base"][a]
        + float(embed @ params["embed_w"][a])
        + float(params["persona_aff"][persona, a])
        + float(params["category_aff"][category, a])
    )
    return float(sigmoid(score))


def sample_conversion(rng: np.random.Generator, p: float) -> int:
    """Sample a binary conversion with probability p."""
    return int(rng.random() < p)

