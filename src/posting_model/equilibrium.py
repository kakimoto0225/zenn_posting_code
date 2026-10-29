from __future__ import annotations

from dataclasses import dataclass

from .bargaining import BargainingSolution, solve_duration
from .config import ModelConfig
from .primitives import State, values


@dataclass(frozen=True)
class Equilibrium:
    outcome: str
    guarantee: float
    posting_fee: float
    player_surplus: float
    mlb_surplus: float
    short: BargainingSolution
    long: BargainingSolution


def solve_equilibrium(state: State, cfg: ModelConfig) -> Equilibrium:
    short = solve_duration("S", values(state, "S", cfg), cfg)
    long = solve_duration("L", values(state, "L", cfg), cfg)
    feasible = [x for x in (short, long) if x.feasible]
    if not feasible:
        return Equilibrium("0", 0.0, 0.0, 0.0, 0.0, short, long)
    chosen = max(feasible, key=lambda x: (x.log_nash, x.duration == "L"))
    return Equilibrium(
        chosen.duration,
        chosen.guarantee,
        chosen.posting_fee,
        chosen.player_surplus,
        chosen.mlb_surplus,
        short,
        long,
    )
