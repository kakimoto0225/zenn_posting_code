from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FeePolicy


@dataclass(frozen=True)
class Segment:
    lower: float
    upper: float
    rate: float
    intercept: float


def segments(policy: FeePolicy) -> tuple[Segment, ...]:
    """最低料金を含む実効的な区分線形料金表を返す。

    `minimum_fee` は従価料金への上乗せではなく、基礎料金の下限として扱う。
    下限と基礎料金の交点は、交渉問題で明示的に比較すべき折れ点になる。
    """
    k1, k2 = policy.kinks
    t1, t2, t3 = policy.rates
    c1 = 0.0
    c2 = c1 + (t1 - t2) * k1
    c3 = c2 + (t2 - t3) * k2
    base = (
        Segment(0.0, k1, t1, c1),
        Segment(k1, k2, t2, c2),
        Segment(k2, np.inf, t3, c3),
    )
    if policy.minimum_fee <= 0.0:
        return base

    effective: list[Segment] = []
    for segment in base:
        value_at_lower = segment.rate * segment.lower + segment.intercept
        if value_at_lower >= policy.minimum_fee:
            effective.append(segment)
            continue
        if segment.rate <= 0.0:
            effective.append(
                Segment(
                    segment.lower,
                    segment.upper,
                    0.0,
                    policy.minimum_fee,
                )
            )
            continue
        crossing = (policy.minimum_fee - segment.intercept) / segment.rate
        if crossing >= segment.upper:
            effective.append(
                Segment(
                    segment.lower,
                    segment.upper,
                    0.0,
                    policy.minimum_fee,
                )
            )
            continue
        effective.append(
            Segment(segment.lower, crossing, 0.0, policy.minimum_fee)
        )
        effective.append(
            Segment(crossing, segment.upper, segment.rate, segment.intercept)
        )
    return tuple(effective)


def breakpoints(policy: FeePolicy) -> tuple[float, ...]:
    """交渉solverが比較すべき有限の料金折れ点を返す。"""
    points = {
        segment.lower
        for segment in segments(policy)
        if np.isfinite(segment.lower) and segment.lower > 0.0
    }
    return tuple(sorted(points))


def fee(guarantee: float | np.ndarray, policy: FeePolicy) -> float | np.ndarray:
    g = np.asarray(guarantee, dtype=float)
    out = np.empty_like(g)
    for segment in segments(policy):
        mask = (g >= segment.lower) & (g <= segment.upper)
        out[mask] = segment.rate * g[mask] + segment.intercept
    out = np.maximum(out, policy.minimum_fee)
    return float(out) if out.ndim == 0 else out
