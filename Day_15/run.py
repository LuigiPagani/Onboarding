from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Allow running from repo root or Day_15 directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Day_15 import agents
from Day_15.ope import dr_estimate, fit_reward_model, predict_all_actions
from Day_15.sim import (
    SimConfig,
    build_conversion_params,
    conversion_probability,
    context_features,
    make_context_matrix,
    model_cost,
    sample_context,
    sample_conversion,
    sample_delay,
)

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


@dataclass
class RunMetrics:
    """Per-round metrics collected from a run."""
    conversion: np.ndarray
    profit: np.ndarray
    unsafe: np.ndarray


@dataclass
class SummaryMetrics:
    """Aggregate metrics for one agent run."""
    dr_conversion: float
    dr_profit: float
    actual_conversion: float
    actual_profit: float
    actual_unsafe: float


def _run_logging_phase(
    cfg: SimConfig,
    params: Dict[str, np.ndarray],
    rng: np.random.Generator,
    rounds: int,
) -> Dict[str, np.ndarray]:
    """Collect a random-policy log with delayed conversions for OPE."""
    n_arms = cfg.n_arms
    d_base = cfg.embed_dim + cfg.category_count + cfg.persona_count + 1

    contexts = np.zeros((rounds, d_base), dtype=np.float64)
    actions = np.zeros(rounds, dtype=int)
    rewards = np.zeros(rounds, dtype=np.float64)
    propensities = np.full(rounds, 1.0 / n_arms, dtype=np.float64)

    # Pending conversion outcomes for delayed feedback: (deliver_at, idx, conv).
    pending: List[Tuple[int, int, int]] = []

    for t in range(rounds):
        # Deliver any conversions whose delay has elapsed.
        for deliver_t, idx, conv in [p for p in pending if p[0] <= t]:
            rewards[idx] = float(conv)
        pending = [p for p in pending if p[0] > t]

        context = sample_context(rng, cfg)
        base = context_features(context, cfg)
        action = int(rng.integers(0, n_arms))

        unsafe_injected = action == 2 and rng.random() < cfg.unsafe_rate
        p_conv = conversion_probability(context, action, params)
        conv = 0 if unsafe_injected else sample_conversion(rng, p_conv)
        delay = sample_delay(rng, cfg)
        if delay == 0:
            rewards[t] = float(conv)
        else:
            pending.append((t + delay, t, conv))

        contexts[t] = base
        actions[t] = action

    for _, idx, conv in pending:
        rewards[idx] = float(conv)

    return {
        "contexts": contexts,
        "actions": actions,
        "rewards": rewards,
        "propensities": propensities,
    }


def _train_agent(
    agent,
    cfg: SimConfig,
    params: Dict[str, np.ndarray],
    rng: np.random.Generator,
    rounds: int,
) -> RunMetrics:
    """Train one agent with delayed conversion feedback."""
    conversion = np.zeros(rounds, dtype=np.float64)
    profit = np.zeros(rounds, dtype=np.float64)
    unsafe = np.zeros(rounds, dtype=np.float64)

    # Pending conversions with costs: (deliver_at, idx, conv, cost, features).
    pending: List[Tuple[int, int, int, float, np.ndarray]] = []

    for t in range(rounds):
        # Apply delayed conversion + profit once the feedback arrives.
        for deliver_t, idx, conv, cost, features in [p for p in pending if p[0] <= t]:
            profit_val = float(conv) * cfg.value_per_conversion - cost
            conversion[idx] = float(conv)
            profit[idx] = profit_val
            agent.update(features, profit_val)
        pending = [p for p in pending if p[0] > t]

        context = sample_context(rng, cfg)
        X = make_context_matrix(context, cfg)
        Xb = agents.block_features(X)

        action = agents.select_arm(agent, Xb, rng)
        unsafe_injected = action == 2 and rng.random() < cfg.unsafe_rate
        is_unsafe = unsafe_injected
        unsafe[t] = 1.0 if is_unsafe else 0.0

        p_conv = conversion_probability(context, action, params)
        conv = 0 if is_unsafe else sample_conversion(rng, p_conv)
        delay = sample_delay(rng, cfg)
        cost = model_cost(action, cfg)

        if delay == 0:
            profit_val = float(conv) * cfg.value_per_conversion - cost
            conversion[t] = float(conv)
            profit[t] = profit_val
            agent.update(Xb[action], profit_val)
        else:
            pending.append((t + delay, t, conv, cost, Xb[action].copy()))

    for _, idx, conv, cost, _ in pending:
        conversion[idx] = float(conv)
        profit[idx] = float(conv) * cfg.value_per_conversion - cost

    return RunMetrics(conversion=conversion, profit=profit, unsafe=unsafe)


def _eval_policy(
    agent,
    cfg: SimConfig,
    params: Dict[str, np.ndarray],
    rng: np.random.Generator,
    rounds: int,
) -> RunMetrics:
    """Evaluate a learned policy without updating it."""
    conversion = np.zeros(rounds, dtype=np.float64)
    profit = np.zeros(rounds, dtype=np.float64)
    unsafe = np.zeros(rounds, dtype=np.float64)

    for t in range(rounds):
        context = sample_context(rng, cfg)
        X = make_context_matrix(context, cfg)
        Xb = agents.block_features(X)

        action = agents.greedy_arm(agent, Xb)
        unsafe_injected = action == 2 and rng.random() < cfg.unsafe_rate
        is_unsafe = unsafe_injected
        unsafe[t] = 1.0 if is_unsafe else 0.0

        p_conv = conversion_probability(context, action, params)
        conv = 0 if is_unsafe else sample_conversion(rng, p_conv)
        cost = model_cost(action, cfg)

        conversion[t] = float(conv)
        profit[t] = float(conv) * cfg.value_per_conversion - cost

    return RunMetrics(conversion=conversion, profit=profit, unsafe=unsafe)


def _policy_actions_from_contexts(agent, contexts: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Compute greedy actions for a batch of logged contexts."""
    actions = np.zeros(contexts.shape[0], dtype=int)
    for i, base in enumerate(contexts):
        X = np.repeat(base[None, :], cfg.n_arms, axis=0)
        Xb = agents.block_features(X)
        actions[i] = agents.greedy_arm(agent, Xb)
    return actions


def _plot_curves(results: Dict[str, RunMetrics]) -> None:
    """Plot cumulative averages for conversion, profit, and safety."""
    if plt is None:
        print("matplotlib not installed; skipping plots")
        return

    plt.figure(figsize=(8, 4))
    for name, metrics in results.items():
        avg = np.cumsum(metrics.conversion) / (np.arange(metrics.conversion.size) + 1)
        plt.plot(avg, label=name)
    plt.title("Cumulative conversion rate")
    plt.xlabel("round")
    plt.ylabel("conversion rate")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    for name, metrics in results.items():
        avg = np.cumsum(metrics.profit) / (np.arange(metrics.profit.size) + 1)
        plt.plot(avg, label=name)
    plt.title("Cumulative avg profit")
    plt.xlabel("round")
    plt.ylabel("profit")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(8, 4))
    for name, metrics in results.items():
        avg = np.cumsum(metrics.unsafe) / (np.arange(metrics.unsafe.size) + 1)
        plt.plot(avg, label=name)
    plt.title("Cumulative unsafe rate")
    plt.xlabel("round")
    plt.ylabel("unsafe rate")
    plt.legend()
    plt.tight_layout()

    plt.show()


def main() -> None:
    """Parse args and run logging, training, OPE, and evaluation."""
    parser = argparse.ArgumentParser(description="Day 15 capstone runner")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--log-rounds", type=int, default=1000)
    parser.add_argument("--eval-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delay-mean", type=float, default=50.0)

    args = parser.parse_args()

    if args.rounds <= args.log_rounds:
        raise SystemExit("--rounds must be greater than --log-rounds")

    summaries, eval_results = run_experiment(
        rounds=args.rounds,
        log_rounds=args.log_rounds,
        eval_rounds=args.eval_rounds,
        seed=args.seed,
        epsilon=args.epsilon,
        delay_mean=args.delay_mean,
        plot=False,
    )

    for name, summary in summaries.items():
        print(f"\n{name}:")
        print(f"  DR conversion estimate: {summary.dr_conversion:.4f}")
        print(f"  DR profit estimate:     {summary.dr_profit:.4f}")
        print(f"  Eval conversion:        {summary.actual_conversion:.4f}")
        print(f"  Eval profit:            {summary.actual_profit:.4f}")
        print(f"  Eval unsafe rate:       {summary.actual_unsafe:.4f}")

    _plot_curves(eval_results)


def run_experiment(
    *,
    rounds: int,
    log_rounds: int,
    eval_rounds: int,
    seed: int,
    epsilon: float,
    delay_mean: float,
    plot: bool,
) -> tuple[Dict[str, SummaryMetrics], Dict[str, RunMetrics]]:
    """Run a full experiment and return summaries plus eval curves."""
    cfg = SimConfig(delay_mean=delay_mean)
    rng = np.random.default_rng(seed)
    params = build_conversion_params(cfg, rng)

    log_data = _run_logging_phase(
        cfg=cfg,
        params=params,
        rng=np.random.default_rng(seed + 1),
        rounds=log_rounds,
    )

    d_base = cfg.embed_dim + cfg.category_count + cfg.persona_count + 1
    d_block = d_base * cfg.n_arms

    agents_map = {
        "linucb": agents.LinUCBWrapper(d=d_block, alpha=1.0, lam=1.0),
        "thompson": agents.LinearThompson(d=d_block, lam=1.0, sigma=0.5),
        "epsilon_greedy": agents.EpsilonGreedyLinear(d=d_block, epsilon=epsilon, lam=1.0),
    }

    train_results: Dict[str, RunMetrics] = {}
    eval_results: Dict[str, RunMetrics] = {}
    summaries: Dict[str, SummaryMetrics] = {}

    for idx, (name, agent) in enumerate(agents_map.items()):
        rng_train = np.random.default_rng(seed + 10 + idx)
        train_results[name] = _train_agent(
            agent=agent,
            cfg=cfg,
            params=params,
            rng=rng_train,
            rounds=rounds - log_rounds,
        )

        rng_eval = np.random.default_rng(seed + 20 + idx)
        eval_results[name] = _eval_policy(
            agent=agent,
            cfg=cfg,
            params=params,
            rng=rng_eval,
            rounds=eval_rounds,
        )

        target_actions = _policy_actions_from_contexts(agent, log_data["contexts"], cfg)
        # DR OPE: estimate value of the learned policy from the random log.
        model = fit_reward_model(
            log_data["contexts"],
            log_data["actions"],
            log_data["rewards"],
            cfg.n_arms,
        )
        q_hat = predict_all_actions(model, log_data["contexts"], cfg.n_arms)
        dr_conv = dr_estimate(
            log_data["contexts"],
            log_data["actions"],
            log_data["rewards"],
            log_data["propensities"],
            target_actions,
            q_hat,
        )

        avg_cost = float(np.mean([cfg.costs[a] for a in target_actions]))
        dr_profit = dr_conv * cfg.value_per_conversion - avg_cost

        actual_conv = float(np.mean(eval_results[name].conversion))
        actual_profit = float(np.mean(eval_results[name].profit))
        actual_unsafe = float(np.mean(eval_results[name].unsafe))

        summaries[name] = SummaryMetrics(
            dr_conversion=dr_conv,
            dr_profit=dr_profit,
            actual_conversion=actual_conv,
            actual_profit=actual_profit,
            actual_unsafe=actual_unsafe,
        )

    if plot:
        _plot_curves(eval_results)

    return summaries, eval_results


if __name__ == "__main__":
    main()
