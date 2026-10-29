import numpy as np

from posting_model.bargaining import solve_duration
from posting_model.config import FeePolicy, ModelConfig
from posting_model.fee_schedule import fee
from posting_model.primitives import DurationValues


def test_global_solution_handles_kinks_and_participation():
    cfg = ModelConfig()
    for r in (10, 30, 60, 120):
        solution = solve_duration("L", DurationValues(r=r, a=2), cfg)
        if solution.feasible:
            assert solution.guarantee >= 0
            assert solution.player_surplus >= 0
            assert solution.mlb_surplus >= 0


def test_solver_matches_dense_grid_for_random_states():
    rng = np.random.default_rng(20260820)
    policies = (FeePolicy(), FeePolicy(name="minimum_5", minimum_fee=5.0))
    for policy in policies:
        cfg = ModelConfig(policy=policy)
        for _ in range(30):
            values = DurationValues(
                r=float(rng.uniform(5.0, 140.0)),
                a=float(rng.uniform(-5.0, 20.0)),
            )
            solved = solve_duration("L", values, cfg)
            grid = np.linspace(0.0, max(values.r, 0.0), 20_001)
            posting_fee = np.asarray(fee(grid, policy))
            mlb = values.r - grid - posting_fee - values.d_m
            player = grid + values.a + cfg.alpha * posting_fee - values.d_p
            feasible = (mlb >= 0.0) & (player >= 0.0)
            if not feasible.any():
                assert not solved.feasible
                continue
            log_nash = (1.0 - cfg.beta) * np.log(
                np.maximum(mlb[feasible], 1e-12)
            ) + cfg.beta * np.log(np.maximum(player[feasible], 1e-12))
            assert solved.feasible
            assert solved.log_nash >= float(log_nash.max()) - 1e-5
