"""Simulation data-generating process with separate random streams."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from .batch import (
    BatchState,
    evaluate_welfare_batch,
    permission_values_batch,
    solve_equilibrium_batch,
)
from .config import ModelConfig


def integration_draws(cfg: ModelConfig) -> np.ndarray:
    """許可価値積分用に、実現DGPと同じ分布から独立な点を生成する。"""
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 101]))
    draws = rng.standard_normal((cfg.integration_draws, 2))
    return _transform_bargaining_shocks(draws, cfg, rng)


def _transform_bargaining_shocks(
    shocks: np.ndarray,
    cfg: ModelConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """標準正規の基礎乱数を設定された交渉ショック分布へ変換する。"""
    transformed = shocks.copy()
    if cfg.heavy_tails:
        # 2成分で同じ尺度混合変数を共有し、多変量t型の裾を作る。
        transformed /= np.sqrt(rng.chisquare(5, size=(len(shocks), 1)) / 5)
    if cfg.rho_uv:
        transformed[:, 1] = (
            cfg.rho_uv * transformed[:, 0]
            + np.sqrt(1 - cfg.rho_uv**2) * transformed[:, 1]
        )
    return transformed


def simulate_sample(
    cfg: ModelConfig,
    n: int | None = None,
    *,
    permission_bargaining_cfg: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """許可と許可後交渉を順に解いて疑似標本を生成する。

    Args:
        cfg: 実現制度とDGPの設定。
        n: 標本サイズ。省略時は設定値を使う。
        permission_bargaining_cfg: 許可価値だけに使う交渉写像。基準設定を
            渡すと固定期待ベンチマークになり、省略時は完全統合モデルとなる。

    Returns:
        観測可能データと潜在結果を含むtruthデータ。
    """
    n = cfg.n if n is None else n
    public_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 1]))
    permission_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 2]))
    state_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 3]))
    measurement_rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, 4]))
    x = public_rng.standard_normal((n, 3))
    z = _transform_bargaining_shocks(
        state_rng.standard_normal((n, 2)), cfg, state_rng
    )
    draws = integration_draws(cfg)
    value = permission_values_batch(
        x, draws, cfg, bargaining_cfg=permission_bargaining_cfg
    )
    p_permission = (
        (value >= 0.0).astype(float)
        if cfg.sigma_kappa <= 0.0
        else expit(value / cfg.sigma_kappa)
    )
    permission_choice_innovation = permission_rng.logistic(size=n)
    permission_choice_shock = cfg.sigma_kappa * permission_choice_innovation
    permitted = value >= permission_choice_shock
    state = BatchState(
        q=x[:, 0],
        age=x[:, 1],
        control=x[:, 2],
        u=cfg.sigma_u * z[:, 0],
        v=cfg.sigma_v * z[:, 1],
    )
    equilibrium = solve_equilibrium_batch(state, cfg)
    welfare = evaluate_welfare_batch(state, equilibrium, cfg)
    if cfg.policy.name == "pullback":
        permitted &= welfare.origin >= 0.0
    signed = permitted & (equilibrium.outcome != "0")
    observed = pd.DataFrame(
        {
            "id": np.arange(n),
            "q": state.q,
            "age": state.age,
            "control": state.control,
            "permission": permitted.astype(int),
            "permission_probability": p_permission,
            "permission_value": value,
            "outcome": np.where(permitted, equilibrium.outcome, "D"),
            "guarantee": np.where(
                signed,
                np.maximum(
                    0.0,
                    equilibrium.guarantee
                    + measurement_rng.normal(0, cfg.guarantee_error, size=n),
                ),
                np.nan,
            ),
            "posting_fee": np.where(signed, equilibrium.posting_fee, np.nan),
        }
    )
    truth = pd.DataFrame(
        {
            "id": np.arange(n),
            "u": state.u,
            "v": state.v,
            "permission_choice_innovation": permission_choice_innovation,
            "permission_choice_shock": permission_choice_shock,
            "potential_outcome": equilibrium.outcome,
            "potential_guarantee": equilibrium.guarantee,
            "potential_fee": equilibrium.posting_fee,
            "origin_welfare": welfare.origin,
            "player_welfare": welfare.player,
            "mlb_welfare": welfare.mlb,
            "total_welfare": welfare.total,
            "excess_short": welfare.excess_short.astype(int),
            "origin_loss": welfare.origin_loss.astype(int),
            "joint_adverse": welfare.joint_adverse.astype(int),
            "potential_origin_loss": welfare.origin_loss.astype(int),
            "potential_no_deal_loss": welfare.no_deal_loss.astype(int),
            "potential_short_contract_loss": welfare.short_contract_loss.astype(int),
            "potential_long_contract_loss": welfare.long_contract_loss.astype(int),
            "potential_excess_short_loss": welfare.excess_short_loss.astype(int),
            "realized_origin_welfare": np.where(permitted, welfare.origin, 0.0),
            "realized_player_welfare": np.where(permitted, welfare.player, 0.0),
            "realized_mlb_welfare": np.where(permitted, welfare.mlb, 0.0),
            "realized_total_welfare": np.where(permitted, welfare.total, 0.0),
            "realized_excess_short": (permitted & welfare.excess_short).astype(int),
            "realized_origin_loss": (permitted & welfare.origin_loss).astype(int),
            "realized_joint_adverse": (permitted & welfare.joint_adverse).astype(int),
            "realized_no_deal_loss": (permitted & welfare.no_deal_loss).astype(int),
            "realized_short_contract_loss": (
                permitted & welfare.short_contract_loss
            ).astype(int),
            "realized_long_contract_loss": (
                permitted & welfare.long_contract_loss
            ).astype(int),
            "realized_excess_short_loss": (
                permitted & welfare.excess_short_loss
            ).astype(int),
        }
    )
    return observed, truth
