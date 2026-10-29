import numpy as np

from posting_model.batch import (
    BatchState,
    evaluate_welfare_batch,
    permission_values_batch,
    solve_equilibrium_batch,
)
from posting_model.config import ModelConfig
from posting_model.equilibrium import solve_equilibrium
from posting_model.permission import permission_value
from posting_model.primitives import State
from posting_model.welfare import evaluate_welfare


def test_batch_solver_matches_scalar_solver() -> None:
    cfg = ModelConfig()
    states = [
        State(q, age, control, u, v)
        for q, age, control, u, v in [
            (-1.0, 0.5, -0.2, -20.0, -2.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, -0.5, 0.2, 20.0, 2.0),
            (0.3, 1.2, -1.0, 8.0, -0.7),
        ]
    ]
    batch_state = BatchState(
        q=np.array([state.q for state in states]),
        age=np.array([state.age for state in states]),
        control=np.array([state.control for state in states]),
        u=np.array([state.u for state in states]),
        v=np.array([state.v for state in states]),
    )
    batch_eq = solve_equilibrium_batch(batch_state, cfg)
    batch_welfare = evaluate_welfare_batch(batch_state, batch_eq, cfg)
    for index, state in enumerate(states):
        scalar_eq = solve_equilibrium(state, cfg)
        scalar_welfare = evaluate_welfare(state, scalar_eq, cfg)
        assert batch_eq.outcome[index] == scalar_eq.outcome
        assert np.isclose(batch_eq.guarantee[index], scalar_eq.guarantee)
        assert np.isclose(batch_eq.posting_fee[index], scalar_eq.posting_fee)
        assert np.isclose(batch_eq.player_surplus[index], scalar_eq.player_surplus)
        assert np.isclose(batch_eq.mlb_surplus[index], scalar_eq.mlb_surplus)
        assert np.isclose(batch_welfare.origin[index], scalar_welfare.origin)
        assert np.isclose(batch_welfare.total[index], scalar_welfare.total)
        assert batch_welfare.excess_short[index] == scalar_welfare.excess_short


def test_batch_permission_value_matches_scalar_value() -> None:
    cfg = ModelConfig(sigma_u=2.0, sigma_v=1.0)
    x = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.2]])
    draws = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    batch = permission_values_batch(x, draws, cfg)
    scalar = np.array(
        [permission_value(q, age, control, draws, cfg) for q, age, control in x]
    )
    assert np.allclose(batch, scalar)
