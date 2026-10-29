from posting_model.config import ModelConfig
from posting_model.equilibrium import solve_equilibrium
from posting_model.primitives import State, values
from posting_model.welfare import evaluate_welfare


def test_transfers_cancel_from_total_welfare():
    cfg = ModelConfig(release_benefit=3)
    state = State(0, 0, 0, 5, 0)
    eq = solve_equilibrium(state, cfg)
    welfare = evaluate_welfare(state, eq, cfg)
    if eq.outcome != "0":
        primitive = values(state, eq.outcome, cfg)
        opportunity = cfg.gamma_n0 + cfg.gamma_nq * state.q + state.control
        assert abs(welfare.total - (primitive.r + primitive.a + cfg.release_benefit - opportunity)) < 1e-8


def test_origin_loss_components_are_disjoint_and_exhaustive():
    cfg = ModelConfig()
    for u in (-50.0, 0.0, 50.0):
        state = State(0.0, 0.0, 0.0, u, 0.0)
        welfare = evaluate_welfare(state, solve_equilibrium(state, cfg), cfg)
        components = [
            welfare.no_deal_loss,
            welfare.short_contract_loss,
            welfare.long_contract_loss,
        ]
        assert sum(components) == int(welfare.origin_loss)
        assert welfare.excess_short_loss == (welfare.excess_short and welfare.origin_loss)
