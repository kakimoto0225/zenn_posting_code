from posting_model.config import ModelConfig
from posting_model.equilibrium import solve_equilibrium
from posting_model.primitives import State


def test_equilibrium_has_exactly_one_outcome():
    cfg = ModelConfig()
    for u in (-30, 0, 30):
        for v in (-3, 0, 3):
            assert solve_equilibrium(State(0, 0, 0, u, v), cfg).outcome in {"0", "S", "L"}
