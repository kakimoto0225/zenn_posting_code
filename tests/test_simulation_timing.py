import numpy as np

from posting_model.config import ModelConfig
from posting_model.simulation import integration_draws, simulate_sample


def test_simulation_is_reproducible_and_hides_denied_states():
    cfg = ModelConfig(n=30, integration_draws=8)
    observed1, truth1 = simulate_sample(cfg)
    observed2, truth2 = simulate_sample(cfg)
    assert observed1.equals(observed2)
    assert truth1.equals(truth2)
    assert observed1.loc[observed1.outcome.eq("D"), "guarantee"].isna().all()
    assert truth1["u"].corr(truth1["permission_choice_innovation"]) < 0.99


def test_realized_post_permission_state_does_not_enter_permission():
    cfg = ModelConfig(n=40, integration_draws=12)
    observed1, _ = simulate_sample(cfg)
    observed2, _ = simulate_sample(cfg.replace(rho_uv=0.7, heavy_tails=True))
    assert not observed1["permission_value"].equals(observed2["permission_value"])


def test_integration_draws_match_alternative_dgp():
    correlated = integration_draws(
        ModelConfig(integration_draws=50_000, rho_uv=0.7)
    )
    assert np.isclose(np.corrcoef(correlated.T)[0, 1], 0.7, atol=0.02)

    normal = integration_draws(ModelConfig(integration_draws=50_000))
    heavy = integration_draws(
        ModelConfig(integration_draws=50_000, heavy_tails=True)
    )
    normal_tail = np.mean(np.abs(normal[:, 0]) > 3.0)
    heavy_tail = np.mean(np.abs(heavy[:, 0]) > 3.0)
    assert heavy_tail > 3.0 * normal_tail


def test_observed_guarantees_are_nonnegative():
    observed, _ = simulate_sample(
        ModelConfig(n=500, integration_draws=16, guarantee_error=100.0)
    )
    assert observed["guarantee"].dropna().ge(0.0).all()


def test_fixed_expectation_and_integrated_models_share_random_streams():
    baseline = ModelConfig(n=40, integration_draws=12)
    counterfactual = baseline.replace(
        policy=baseline.policy.__class__(name="flat_40", rates=(0.4, 0.4, 0.4))
    )
    integrated, integrated_truth = simulate_sample(counterfactual)
    fixed, fixed_truth = simulate_sample(
        counterfactual,
        permission_bargaining_cfg=baseline,
    )
    assert integrated_truth[["u", "v", "permission_choice_shock"]].equals(
        fixed_truth[["u", "v", "permission_choice_shock"]]
    )
    assert not integrated["permission_value"].equals(fixed["permission_value"])
