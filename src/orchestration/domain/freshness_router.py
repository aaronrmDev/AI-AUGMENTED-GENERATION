"""Concept 9's Freshness-Aware Data Router, as pure rules: where a source lives, how
long a cached copy is trusted, and when observed change should move it."""
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta

from src.orchestration.domain.entities import (
    DataSource,
    DataSourceProfile,
    FreshnessPolicy,
    IngestionRoute,
    MigrationDecision,
    SourceScope,
)

_SOURCE_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_OID, "orchestration:data_source")


def source_id_for(tenant_id: uuid.UUID, source_key: str, user_id: uuid.UUID | None) -> uuid.UUID:
    """Deterministic, so a retried ingestion replaces the same RAG document and cache
    entry instead of orphaning the previous ones under a fresh id."""
    return uuid.uuid5(_SOURCE_NAMESPACE, f"{tenant_id}:{user_id or ''}:{source_key}")


def route_for(profile: DataSourceProfile, policy: FreshnessPolicy) -> IngestionRoute:
    if profile.scope is SourceScope.USER:
        return IngestionRoute.MAG
    if profile.expected_change_interval < policy.volatile_below:
        return IngestionRoute.RAG_ONLY
    return IngestionRoute.CAG_WITH_RAG_BACKUP


def cache_ttl(interval: timedelta, policy: FreshnessPolicy) -> timedelta | None:
    if policy.ttl_factor is None:
        return None
    return interval * policy.ttl_factor


def observed_change_interval(times: Sequence[datetime]) -> timedelta | None:
    if len(times) < 2:
        return None
    ordered = sorted(times)
    return (ordered[-1] - ordered[0]) / (len(ordered) - 1)


def decide_migration(
    source: DataSource,
    version_times: Sequence[datetime],
    now: datetime,
    policy: FreshnessPolicy,
) -> MigrationDecision | None:
    """version_times holds every version's ingestion time, the first one included.

    A cached source is demoted when its last demote_after_changes changes, and the
    version before them, all fall strictly inside that many boundaries before now.
    Their mean gap is then under the boundary, so the re-learned interval routes
    RAG_ONLY. A RAG_ONLY source is promoted once it has gone
    promote_after_quiet_multiple boundaries without a new version. MAG sources
    never migrate.
    """
    if source.route is IngestionRoute.MAG:
        return None
    history = sorted(t for t in version_times if t <= now)
    if not history:
        return None
    boundary = policy.volatile_below
    if source.route is IngestionRoute.CAG_WITH_RAG_BACKUP:
        needed = policy.demote_after_changes + 1
        recent = history[-needed:]
        if len(recent) < needed or recent[0] <= now - boundary * policy.demote_after_changes:
            return None
        interval = observed_change_interval(recent)
        if interval is None:
            return None
        return MigrationDecision(IngestionRoute.RAG_ONLY, interval)
    quiet = now - history[-1]
    if quiet >= boundary * policy.promote_after_quiet_multiple:
        return MigrationDecision(IngestionRoute.CAG_WITH_RAG_BACKUP, quiet)
    return None
