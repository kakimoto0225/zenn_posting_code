from posting_model.bargaining import solve_duration
from posting_model.config import FeePolicy, ModelConfig
from posting_model.primitives import DurationValues


def test_global_solver_matches_single_segment_closed_form():
    cfg = ModelConfig(policy=FeePolicy(rates=(.2, .2, .2)), beta=.5, alpha=0)
    values = DurationValues(r=60, a=8)
    solution = solve_duration("S", values, cfg)
    expected = cfg.beta * values.r / 1.2 + (1 - cfg.beta) * (-values.a)
    assert abs(solution.guarantee - expected) < 1e-8
    assert solution.player_surplus >= 0
    assert solution.mlb_surplus >= 0
