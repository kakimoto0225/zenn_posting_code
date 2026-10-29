from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeePolicy:
    name: str = "current"
    rates: tuple[float, float, float] = (0.20, 0.175, 0.15)
    kinks: tuple[float, float] = (25.0, 50.0)
    trailing_rate: float = 0.0
    minimum_fee: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    run_id: str = "baseline"
    seed: int = 20260812
    n: int = 500
    integration_draws: int = 256
    replications: int = 10
    beta: float = 0.5
    alpha: float = 0.0
    eta_a: float = 1.0
    sigma_u: float = 7.0
    sigma_v: float = 1.0
    sigma_kappa: float = 3.0
    gamma_n0: float = 7.5
    gamma_nq: float = 1.5
    release_benefit: float = 3.0
    no_deal_cost: float = 1.0
    rho_uv: float = 0.0
    heavy_tails: bool = False
    r0: float = 30.0
    r_q: float = 6.0
    r_age: float = -1.5
    delta_r0: float = 18.0
    delta_r_q: float = 2.0
    a0: float = 2.0
    a_q: float = 1.0
    delta_a0: float = 2.0
    delta_a_q: float = 0.25
    delta_a_v: float = 0.35
    guarantee_error: float = 0.5
    policy: FeePolicy = field(default_factory=FeePolicy)

    def replace(self, **kwargs: object) -> ModelConfig:
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["policy"] = asdict(self.policy)
        return result


def load_config(path: str | Path) -> ModelConfig:
    """TOML設定を読み込み、旧パラメータ名を正規化する。

    Args:
        path: 設定ファイルのパス。

    Returns:
        正規化済みのモデル設定。

    Raises:
        ValueError: 新旧の許可費用尺度が同時に指定された場合。
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if "sigma_n" in raw and "sigma_kappa" in raw:
        raise ValueError("Specify only sigma_kappa; sigma_n is a legacy alias")
    if "sigma_n" in raw:
        raw["sigma_kappa"] = raw.pop("sigma_n")
    policy_raw = raw.pop("policy", {})
    if "rates" in policy_raw:
        policy_raw["rates"] = tuple(policy_raw["rates"])
    if "kinks" in policy_raw:
        policy_raw["kinks"] = tuple(policy_raw["kinks"])
    return ModelConfig(**raw, policy=FeePolicy(**policy_raw))
