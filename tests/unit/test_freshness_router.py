import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.orchestration.domain.entities import (
    DataSource,
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    MigrationDecision,
    SourceScope,
)
from src.orchestration.domain.freshness_router import (
    cache_ttl,
    decide_migration,
    observed_change_interval,
    route_for,
    source_id_for,
)

POLICY = FreshnessPolicy()
NOW = datetime(2026, 1, 20, tzinfo=UTC)
HOUR, DAY = timedelta(hours=1), timedelta(days=1)
TENANT, USER = uuid.uuid4(), uuid.uuid4()


def _source(route: IngestionRoute, scope: SourceScope = SourceScope.TENANT) -> DataSource:
    return DataSource(
        id=uuid.uuid4(), tenant_id=TENANT, user_id=USER if scope is SourceScope.USER else None,
        source_key="k", scope=scope, expected_change_interval=DAY, route=route,
        content_hash="h", last_changed_at=NOW, last_ingested_at=NOW,
    )


# Concept 9's freshness spectrum, each row declared as a profile.
@pytest.mark.parametrize(
    ("scope", "interval", "expected"),
    [
        (SourceScope.TENANT, timedelta(seconds=1), IngestionRoute.RAG_ONLY),  # stock prices
        (SourceScope.TENANT, HOUR, IngestionRoute.RAG_ONLY),  # news, social media
        (SourceScope.USER, timedelta(minutes=1), IngestionRoute.MAG),  # session state
        (SourceScope.TENANT, 3 * DAY, IngestionRoute.CAG_WITH_RAG_BACKUP),  # product catalog
        (SourceScope.TENANT, 60 * DAY, IngestionRoute.CAG_WITH_RAG_BACKUP),  # policies
        (SourceScope.TENANT, 3650 * DAY, IngestionRoute.CAG_WITH_RAG_BACKUP),  # textbooks
        # Code repositories change "per commit". One interval can't express the
        # source's main-branch/PR split, so a busy repository routes RAG_ONLY.
        (SourceScope.TENANT, 4 * HOUR, IngestionRoute.RAG_ONLY),
        (SourceScope.USER, 30 * DAY, IngestionRoute.MAG),  # long-term preferences
    ],
)
def test_concept_9_rows_route_as_declared(scope, interval, expected):
    assert route_for(DataSourceProfile("s", scope, interval), POLICY) is expected


def test_exactly_one_boundary_is_stable_and_just_under_is_volatile():
    at = DataSourceProfile("s", SourceScope.TENANT, DAY)
    under = DataSourceProfile("s", SourceScope.TENANT, DAY - timedelta(seconds=1))
    assert route_for(at, POLICY) is IngestionRoute.CAG_WITH_RAG_BACKUP
    assert route_for(under, POLICY) is IngestionRoute.RAG_ONLY


def test_user_scope_routes_to_mag_whatever_its_interval():
    profile = DataSourceProfile("s", SourceScope.USER, 3650 * DAY)
    assert route_for(profile, POLICY) is IngestionRoute.MAG


def test_ttl_is_half_the_interval_by_default_and_none_when_disabled():
    assert cache_ttl(7 * DAY, POLICY) == timedelta(days=3.5)
    assert cache_ttl(7 * DAY, FreshnessPolicy(ttl_factor=None)) is None


def test_observed_interval_needs_two_timestamps_and_averages_uneven_gaps():
    assert observed_change_interval([NOW]) is None
    times = [NOW, NOW + HOUR, NOW + 4 * HOUR]  # gaps of 1h and 3h
    assert observed_change_interval(list(reversed(times))) == 2 * HOUR


def _demote_times(first_gap_before_now: timedelta) -> list[datetime]:
    # Four versions: the one before the last three changes, then three hourly changes.
    start = NOW - first_gap_before_now
    return [start, start + HOUR, start + 2 * HOUR, start + 3 * HOUR]


def test_three_recent_fast_changes_demote_a_cached_source():
    decision = decide_migration(
        _source(IngestionRoute.CAG_WITH_RAG_BACKUP), _demote_times(DAY), NOW, POLICY
    )
    assert decision == MigrationDecision(IngestionRoute.RAG_ONLY, HOUR)


def test_two_changes_are_not_enough_to_demote():
    times = [NOW - 3 * HOUR, NOW - 2 * HOUR, NOW - HOUR]
    source = _source(IngestionRoute.CAG_WITH_RAG_BACKUP)
    assert decide_migration(source, times, NOW, POLICY) is None


def test_a_version_exactly_on_the_window_edge_does_not_demote():
    start = NOW - 3 * DAY
    times = [start, start + DAY, start + 2 * DAY, NOW]  # daily changes: exactly the boundary
    source = _source(IngestionRoute.CAG_WITH_RAG_BACKUP)
    assert decide_migration(source, times, NOW, POLICY) is None


def test_fast_changes_that_ended_before_the_window_do_not_demote():
    times = _demote_times(10 * DAY)  # fast, but ten days ago
    source = _source(IngestionRoute.CAG_WITH_RAG_BACKUP)
    assert decide_migration(source, times, NOW, POLICY) is None


def test_seven_quiet_boundaries_promote_a_rag_only_source_with_the_quiet_span():
    last_change = NOW - 7 * DAY
    decision = decide_migration(
        _source(IngestionRoute.RAG_ONLY), [last_change - HOUR, last_change], NOW, POLICY
    )
    assert decision == MigrationDecision(IngestionRoute.CAG_WITH_RAG_BACKUP, 7 * DAY)


def test_one_second_short_of_seven_quiet_boundaries_does_not_promote():
    last_change = NOW - 7 * DAY + timedelta(seconds=1)
    assert decide_migration(_source(IngestionRoute.RAG_ONLY), [last_change], NOW, POLICY) is None


def test_mag_sources_never_migrate():
    source = _source(IngestionRoute.MAG, SourceScope.USER)
    assert decide_migration(source, _demote_times(DAY), NOW, POLICY) is None
    assert decide_migration(source, [NOW - 30 * DAY], NOW, POLICY) is None


def test_versions_after_now_are_ignored():
    times = [NOW - 8 * DAY, NOW + HOUR]
    decision = decide_migration(_source(IngestionRoute.RAG_ONLY), times, NOW, POLICY)
    assert decision == MigrationDecision(IngestionRoute.CAG_WITH_RAG_BACKUP, 8 * DAY)


def test_source_ids_are_deterministic_and_distinguish_scope():
    tenant_source = source_id_for(TENANT, "prices", None)
    assert tenant_source == source_id_for(TENANT, "prices", None)
    assert tenant_source != source_id_for(TENANT, "prices", USER)
    assert tenant_source != source_id_for(uuid.uuid4(), "prices", None)


def test_a_burst_of_simultaneous_changes_demotes_with_the_minimum_positive_interval():
    times = [NOW - HOUR] * 4  # four versions ingested under one timestamp
    decision = decide_migration(_source(IngestionRoute.CAG_WITH_RAG_BACKUP), times, NOW, POLICY)
    assert decision == MigrationDecision(IngestionRoute.RAG_ONLY, timedelta(seconds=1))
