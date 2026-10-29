from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig
from .equilibrium import Equilibrium
from .fee_schedule import fee
from .primitives import State, values


@dataclass(frozen=True)
class Welfare:
    origin: float
    player: float
    mlb: float
    total: float
    excess_short: bool
    origin_loss: bool
    joint_adverse: bool
    no_deal_loss: bool
    short_contract_loss: bool
    long_contract_loss: bool
    excess_short_loss: bool


def material_surplus(state: State, duration: str, cfg: ModelConfig) -> float:
    """移転を除いた三者物質的余剰を返す。"""
    v = values(state, duration, cfg)
    gross_a = v.a / max(1.0 - cfg.policy.trailing_rate, 1e-8)
    return v.r + gross_a - v.d_m - v.d_p + cfg.release_benefit


def origin_payoff(
    state: State,
    eq: Equilibrium,
    cfg: ModelConfig,
    *,
    recompute_fee: bool = False,
) -> float:
    """均衡契約から移籍元球団の実現利得を計算する。

    Args:
        state: 公開状態と許可後状態。
        eq: 許可後交渉の解。
        cfg: 利得評価に使う制度設定。
        recompute_fee: 保証額を ``cfg`` の料率で再評価するか。

    Returns:
        移籍元球団の実現利得。
    """
    if eq.outcome == "0":
        return -cfg.no_deal_cost
    opportunity = cfg.gamma_n0 + cfg.gamma_nq * state.q + state.control
    net_continuation = values(state, eq.outcome, cfg).a
    gross_continuation = net_continuation / max(1.0 - cfg.policy.trailing_rate, 1e-8)
    trailing = cfg.policy.trailing_rate * gross_continuation
    posting_fee = float(fee(eq.guarantee, cfg.policy)) if recompute_fee else eq.posting_fee
    return posting_fee + trailing + cfg.release_benefit - opportunity


def evaluate_welfare(state: State, eq: Equilibrium, cfg: ModelConfig) -> Welfare:
    """交渉解から三者利得と事後的厚生分類を計算する。"""
    if eq.outcome == "0":
        origin = -cfg.no_deal_cost
        return Welfare(origin, 0.0, 0.0, origin, False, True, False, True, False, False, False)
    origin = origin_payoff(state, eq, cfg)
    player = eq.player_surplus - cfg.alpha * eq.posting_fee
    mlb = eq.mlb_surplus
    total = origin + player + mlb
    excess = eq.outcome == "S" and material_surplus(state, "L", cfg) > material_surplus(state, "S", cfg) + 1e-9
    loss = origin < 0.0
    short_loss = eq.outcome == "S" and loss
    long_loss = eq.outcome == "L" and loss
    return Welfare(
        origin,
        player,
        mlb,
        total,
        excess,
        loss,
        excess and loss,
        False,
        short_loss,
        long_loss,
        excess and loss,
    )
