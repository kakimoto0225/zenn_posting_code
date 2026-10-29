from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig


@dataclass(frozen=True)
class State:
    q: float
    age: float
    control: float
    u: float
    v: float


@dataclass(frozen=True)
class DurationValues:
    r: float
    a: float
    d_m: float = 0.0
    d_p: float = 0.0


def values(state: State, duration: str, cfg: ModelConfig) -> DurationValues:
    base_r = cfg.r0 + cfg.r_q * state.q + cfg.r_age * state.age + state.u
    a_long_gross = cfg.a0 + cfg.a_q * state.q + 0.25 * state.u
    if duration == "S":
        delta_a = np.exp(
            cfg.delta_a0 + cfg.delta_a_q * state.q + cfg.delta_a_v * state.v
        ) * cfg.eta_a
        gross = a_long_gross + delta_a
        return DurationValues(r=base_r + 2.0 * state.v, a=gross * (1.0 - cfg.policy.trailing_rate))
    if duration == "L":
        delta_r = cfg.delta_r0 + cfg.delta_r_q * state.q - 2.0 * state.v
        return DurationValues(r=base_r + delta_r, a=a_long_gross * (1.0 - cfg.policy.trailing_rate))
    raise ValueError(f"Unknown duration: {duration}")
