import numpy as np

from posting_model.config import FeePolicy
from posting_model.fee_schedule import breakpoints, fee


def test_fee_schedule_is_continuous_at_kinks():
    policy = FeePolicy()
    for kink in policy.kinks:
        assert np.isclose(fee(kink - 1e-8, policy), fee(kink + 1e-8, policy), atol=1e-7)


def test_no_fee_policy_is_zero():
    policy = FeePolicy(rates=(0, 0, 0))
    assert np.allclose(fee(np.array([0, 25, 50, 100]), policy), 0)


def test_minimum_fee_is_a_floor_not_an_additive_charge():
    policy = FeePolicy(name="minimum_5", minimum_fee=5.0)
    guarantees = np.array([0.0, 10.0, 25.0, 50.0, 100.0])
    assert np.allclose(fee(guarantees, policy), [5.0, 5.0, 5.0, 9.375, 16.875])
    assert 25.0 in breakpoints(policy)
