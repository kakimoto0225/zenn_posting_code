from __future__ import annotations

import numpy as np
from scipy.special import expit

from .config import ModelConfig
from .equilibrium import solve_equilibrium
from .primitives import State
from .welfare import origin_payoff


def permission_value(
    q: float,
    age: float,
    control: float,
    draws: np.ndarray,
    cfg: ModelConfig,
    *,
    bargaining_cfg: ModelConfig | None = None,
) -> float:
    """許可後交渉の均衡利得を数値積分する。

    Args:
        q: 選手品質。
        age: 年齢状態。
        control: 移籍元球団の保有・戦力事情。
        draws: 許可後ショックの標準化積分点。
        cfg: 許可後利得を評価する制度設定。
        bargaining_cfg: 交渉写像に使う設定。省略時は ``cfg`` と同じ。

    Returns:
        移籍元球団の条件付き期待均衡利得。
    """
    solution_cfg = cfg if bargaining_cfg is None else bargaining_cfg
    payoffs = []
    for u0, v0 in draws:
        state = State(q, age, control, cfg.sigma_u * u0, cfg.sigma_v * v0)
        equilibrium = solve_equilibrium(state, solution_cfg)
        payoffs.append(
            origin_payoff(
                state,
                equilibrium,
                cfg,
                recompute_fee=bargaining_cfg is not None,
            )
        )
    return float(np.mean(payoffs))


def permission_probability(value: float, cfg: ModelConfig) -> float:
    """研究者から未観察の許可費用を積分した許可確率を返す。"""
    if cfg.sigma_kappa <= 0.0:
        return float(value >= 0.0)
    return float(expit(value / cfg.sigma_kappa))
