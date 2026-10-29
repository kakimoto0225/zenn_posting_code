from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ModelConfig
from .fee_schedule import breakpoints, fee, segments
from .primitives import DurationValues


@dataclass(frozen=True)
class BargainingSolution:
    feasible: bool
    duration: str
    guarantee: float
    posting_fee: float
    player_surplus: float
    mlb_surplus: float
    log_nash: float


def _candidate(g: float, duration: str, v: DurationValues, cfg: ModelConfig) -> BargainingSolution:
    f = float(fee(g, cfg.policy))
    sm = v.r - g - f - v.d_m
    sp = g + v.a + cfg.alpha * f - v.d_p
    feasible = g >= 0.0 and sm >= -1e-9 and sp >= -1e-9
    if not feasible:
        return BargainingSolution(False, duration, g, f, sp, sm, -math.inf)
    sm = max(sm, 1e-12)
    sp = max(sp, 1e-12)
    objective = (1.0 - cfg.beta) * math.log(sm) + cfg.beta * math.log(sp)
    return BargainingSolution(True, duration, g, f, sp, sm, objective)


def solve_duration(duration: str, v: DurationValues, cfg: ModelConfig) -> BargainingSolution:
    candidates: list[float] = [0.0, *breakpoints(cfg.policy)]
    for segment in segments(cfg.policy):
        tau, c = segment.rate, segment.intercept
        g_star = (
            cfg.beta * (v.r - c - v.d_m) / (1.0 + tau)
            + (1.0 - cfg.beta) * (v.d_p - v.a - cfg.alpha * c) / (1.0 + cfg.alpha * tau)
        )
        candidates.append(min(max(g_star, segment.lower), segment.upper))
        upper_participation = (v.r - c - v.d_m) / (1.0 + tau)
        if upper_participation >= segment.lower:
            candidates.append(min(upper_participation, segment.upper))
    unique = sorted({round(x, 12) for x in candidates if math.isfinite(x) and x >= 0.0})
    solved = [_candidate(g, duration, v, cfg) for g in unique]
    feasible = [x for x in solved if x.feasible]
    if not feasible:
        return BargainingSolution(False, duration, math.nan, math.nan, math.nan, math.nan, -math.inf)
    return max(feasible, key=lambda x: (x.log_nash, x.guarantee, x.duration == "L"))
