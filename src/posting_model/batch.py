"""Vectorized equilibrium and welfare calculations for simulation experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .fee_schedule import breakpoints, fee, segments


@dataclass(frozen=True)
class BatchState:
    q: np.ndarray
    age: np.ndarray
    control: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class BatchEquilibrium:
    outcome: np.ndarray
    guarantee: np.ndarray
    posting_fee: np.ndarray
    player_surplus: np.ndarray
    mlb_surplus: np.ndarray
    log_nash: np.ndarray


@dataclass(frozen=True)
class BatchWelfare:
    origin: np.ndarray
    player: np.ndarray
    mlb: np.ndarray
    total: np.ndarray
    excess_short: np.ndarray
    origin_loss: np.ndarray
    joint_adverse: np.ndarray
    no_deal_loss: np.ndarray
    short_contract_loss: np.ndarray
    long_contract_loss: np.ndarray
    excess_short_loss: np.ndarray


def duration_values(
    state: BatchState, duration: str, cfg: ModelConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return destination value and net player continuation value."""
    base_r = cfg.r0 + cfg.r_q * state.q + cfg.r_age * state.age + state.u
    a_long_gross = cfg.a0 + cfg.a_q * state.q + 0.25 * state.u
    if duration == "S":
        delta_a = (
            np.exp(cfg.delta_a0 + cfg.delta_a_q * state.q + cfg.delta_a_v * state.v)
            * cfg.eta_a
        )
        gross = a_long_gross + delta_a
        return base_r + 2.0 * state.v, gross * (1.0 - cfg.policy.trailing_rate)
    if duration == "L":
        delta_r = cfg.delta_r0 + cfg.delta_r_q * state.q - 2.0 * state.v
        return base_r + delta_r, a_long_gross * (1.0 - cfg.policy.trailing_rate)
    raise ValueError(f"Unknown duration: {duration}")


def _solve_duration_batch(
    state: BatchState, duration: str, cfg: ModelConfig
) -> BatchEquilibrium:
    r, a = duration_values(state, duration, cfg)
    n = len(r)
    candidate_columns = [np.zeros(n)]
    candidate_columns.extend(np.full(n, point) for point in breakpoints(cfg.policy))
    for segment in segments(cfg.policy):
        tau, c = segment.rate, segment.intercept
        g_star = cfg.beta * (r - c) / (1.0 + tau) + (1.0 - cfg.beta) * (
            -a - cfg.alpha * c
        ) / (1.0 + cfg.alpha * tau)
        candidate_columns.append(np.clip(g_star, segment.lower, segment.upper))
        upper_participation = (r - c) / (1.0 + tau)
        candidate_columns.append(
            np.where(
                upper_participation >= segment.lower,
                np.clip(upper_participation, segment.lower, segment.upper),
                np.nan,
            )
        )

    guarantees = np.column_stack(candidate_columns)
    posting_fees = np.asarray(fee(guarantees, cfg.policy), dtype=float)
    mlb_surplus = r[:, None] - guarantees - posting_fees
    player_surplus = guarantees + a[:, None] + cfg.alpha * posting_fees
    feasible = (
        np.isfinite(guarantees)
        & (guarantees >= 0.0)
        & (mlb_surplus >= -1e-9)
        & (player_surplus >= -1e-9)
    )
    safe_mlb = np.maximum(mlb_surplus, 1e-12)
    safe_player = np.maximum(player_surplus, 1e-12)
    log_nash = (1.0 - cfg.beta) * np.log(safe_mlb) + cfg.beta * np.log(safe_player)
    log_nash = np.where(feasible, log_nash, -np.inf)

    # Scalar solver tie-break: maximize log Nash, then the guarantee.
    best = np.zeros(n, dtype=int)
    for column in range(1, guarantees.shape[1]):
        current = best
        improve = (log_nash[:, column] > log_nash[np.arange(n), current]) | (
            (log_nash[:, column] == log_nash[np.arange(n), current])
            & (guarantees[:, column] > guarantees[np.arange(n), current])
        )
        best = np.where(improve, column, current)
    row = np.arange(n)
    best_log = log_nash[row, best]
    solved = np.isfinite(best_log)
    return BatchEquilibrium(
        outcome=np.where(solved, duration, "0"),
        guarantee=np.where(solved, guarantees[row, best], np.nan),
        posting_fee=np.where(solved, posting_fees[row, best], np.nan),
        player_surplus=np.where(solved, player_surplus[row, best], np.nan),
        mlb_surplus=np.where(solved, mlb_surplus[row, best], np.nan),
        log_nash=best_log,
    )


def solve_equilibrium_batch(state: BatchState, cfg: ModelConfig) -> BatchEquilibrium:
    """Solve short and long contracts globally for all states."""
    short = _solve_duration_batch(state, "S", cfg)
    long = _solve_duration_batch(state, "L", cfg)
    # Scalar solver tie-break selects long duration when Nash products are equal.
    choose_long = np.isfinite(long.log_nash) & (
        ~np.isfinite(short.log_nash) | (long.log_nash >= short.log_nash)
    )
    has_deal = np.isfinite(short.log_nash) | np.isfinite(long.log_nash)

    def choose(
        short_value: np.ndarray, long_value: np.ndarray, no_deal: float
    ) -> np.ndarray:
        return np.where(
            has_deal, np.where(choose_long, long_value, short_value), no_deal
        )

    return BatchEquilibrium(
        outcome=np.where(has_deal, np.where(choose_long, "L", "S"), "0"),
        guarantee=choose(short.guarantee, long.guarantee, 0.0),
        posting_fee=choose(short.posting_fee, long.posting_fee, 0.0),
        player_surplus=choose(short.player_surplus, long.player_surplus, 0.0),
        mlb_surplus=choose(short.mlb_surplus, long.mlb_surplus, 0.0),
        log_nash=choose(short.log_nash, long.log_nash, -np.inf),
    )


def origin_payoff_batch(
    state: BatchState,
    equilibrium: BatchEquilibrium,
    cfg: ModelConfig,
    *,
    recompute_fee: bool = False,
) -> np.ndarray:
    """Evaluate origin-club payoff for a vector of equilibrium outcomes."""
    opportunity = cfg.gamma_n0 + cfg.gamma_nq * state.q + state.control
    _, short_a = duration_values(state, "S", cfg)
    _, long_a = duration_values(state, "L", cfg)
    net_continuation = np.where(equilibrium.outcome == "S", short_a, long_a)
    gross_continuation = net_continuation / max(1.0 - cfg.policy.trailing_rate, 1e-8)
    trailing = cfg.policy.trailing_rate * gross_continuation
    posting_fee = (
        np.asarray(fee(equilibrium.guarantee, cfg.policy), dtype=float)
        if recompute_fee
        else equilibrium.posting_fee
    )
    deal_payoff = posting_fee + trailing + cfg.release_benefit - opportunity
    return np.where(equilibrium.outcome == "0", -cfg.no_deal_cost, deal_payoff)


def evaluate_welfare_batch(
    state: BatchState, equilibrium: BatchEquilibrium, cfg: ModelConfig
) -> BatchWelfare:
    """Calculate realized material welfare and mutually exclusive loss categories."""
    no_deal = equilibrium.outcome == "0"
    origin = origin_payoff_batch(state, equilibrium, cfg)
    player = np.where(
        no_deal,
        0.0,
        equilibrium.player_surplus - cfg.alpha * equilibrium.posting_fee,
    )
    mlb = np.where(no_deal, 0.0, equilibrium.mlb_surplus)
    total = origin + player + mlb
    short_r, short_a = duration_values(state, "S", cfg)
    long_r, long_a = duration_values(state, "L", cfg)
    short_material = (
        short_r
        + short_a / max(1.0 - cfg.policy.trailing_rate, 1e-8)
        + cfg.release_benefit
    )
    long_material = (
        long_r
        + long_a / max(1.0 - cfg.policy.trailing_rate, 1e-8)
        + cfg.release_benefit
    )
    excess_short = (equilibrium.outcome == "S") & (
        long_material > short_material + 1e-9
    )
    origin_loss = origin < 0.0
    short_loss = (equilibrium.outcome == "S") & origin_loss
    long_loss = (equilibrium.outcome == "L") & origin_loss
    excess_short_loss = excess_short & origin_loss
    return BatchWelfare(
        origin=origin,
        player=player,
        mlb=mlb,
        total=total,
        excess_short=excess_short,
        origin_loss=origin_loss,
        joint_adverse=excess_short_loss,
        no_deal_loss=no_deal,
        short_contract_loss=short_loss,
        long_contract_loss=long_loss,
        excess_short_loss=excess_short_loss,
    )


def permission_values_batch(
    x: np.ndarray,
    draws: np.ndarray,
    cfg: ModelConfig,
    *,
    bargaining_cfg: ModelConfig | None = None,
) -> np.ndarray:
    """Integrate equilibrium origin payoffs over post-permission shocks."""
    n, integration_n = len(x), len(draws)
    solution_cfg = cfg if bargaining_cfg is None else bargaining_cfg
    # Bound peak memory for publication-size policy samples without changing
    # the integration points or their order.
    block_size = max(1, 200_000 // integration_n)
    values = np.empty(n, dtype=float)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = x[start:stop]
        repeated = np.repeat(block, integration_n, axis=0)
        tiled = np.tile(draws, (len(block), 1))
        state = BatchState(
            q=repeated[:, 0],
            age=repeated[:, 1],
            control=repeated[:, 2],
            u=cfg.sigma_u * tiled[:, 0],
            v=cfg.sigma_v * tiled[:, 1],
        )
        equilibrium = solve_equilibrium_batch(state, solution_cfg)
        payoff = origin_payoff_batch(
            state, equilibrium, cfg, recompute_fee=bargaining_cfg is not None
        )
        values[start:stop] = payoff.reshape(len(block), integration_n).mean(axis=1)
    return values
