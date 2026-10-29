import numpy as np

from posting_model.config import ModelConfig
from posting_model.equilibrium import solve_equilibrium
from posting_model.permission import permission_probability, permission_value
from posting_model.primitives import State
from posting_model.welfare import evaluate_welfare


def test_permission_value_equals_average_origin_payoff():
    cfg = ModelConfig(sigma_u=2, sigma_v=1)
    draws = np.array([[-1, 0], [0, 0], [1, 0]], dtype=float)
    expected = np.mean([evaluate_welfare(State(0, 0, 0, cfg.sigma_u * u, cfg.sigma_v * v), solve_equilibrium(State(0, 0, 0, cfg.sigma_u * u, cfg.sigma_v * v), cfg), cfg).origin for u, v in draws])
    assert np.isclose(permission_value(0, 0, 0, draws, cfg), expected)


def test_deterministic_permission_when_cost_dispersion_is_zero():
    cfg = ModelConfig(sigma_kappa=0.0)
    assert permission_probability(0.0, cfg) == 1.0
    assert permission_probability(-1e-9, cfg) == 0.0
