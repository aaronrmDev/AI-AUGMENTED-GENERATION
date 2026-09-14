from datetime import timedelta

import pytest

from src.orchestration.domain.entities import DataSourceProfile, FreshnessPolicy, SourceScope


def test_a_blank_source_key_is_rejected():
    with pytest.raises(ValueError):
        DataSourceProfile("  ", SourceScope.TENANT, timedelta(hours=1))


def test_a_non_positive_change_interval_is_rejected():
    with pytest.raises(ValueError):
        DataSourceProfile("prices", SourceScope.TENANT, timedelta(0))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"volatile_below": timedelta(0)},
        {"demote_after_changes": 0},
        {"demote_after_changes": 3, "promote_after_quiet_multiple": 3},
        {"ttl_factor": 0.0},
    ],
)
def test_an_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        FreshnessPolicy(**kwargs)


def test_the_default_policy_matches_the_spec():
    policy = FreshnessPolicy()
    assert policy.volatile_below == timedelta(days=1)
    assert (policy.demote_after_changes, policy.promote_after_quiet_multiple) == (3, 7)
    assert policy.ttl_factor == 0.5


def test_ttl_can_be_disabled():
    assert FreshnessPolicy(ttl_factor=None).ttl_factor is None
