"""The one definition of "does this tenant/user own this ingestion job" --
CeleryIngestionJobDispatcher (a Redis string value) and FakeIngestionJobDispatcher (an
in-memory stand-in) both store and check ownership through these two functions instead
of each re-implementing the same rule."""
import uuid


def owner_key(tenant_id: uuid.UUID, user_id: uuid.UUID | None) -> str:
    """The stored form of "this job belongs to this tenant (and, for a user-scoped
    source, to this user)". A tenant-scoped job carries no user_id, so its key ends in
    a bare trailing colon -- that empty segment is what owner_matches() below treats as
    "visible to any user in the tenant" rather than "visible to no one"."""
    return f"{tenant_id}:{user_id if user_id is not None else ''}"


def owner_matches(stored: str, *, tenant_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    """True when tenant_id/user_id may see the job owner_key() produced `stored` for.
    A tenant-scoped job (no user recorded) is visible to any user in that tenant; a
    user-scoped job is visible only to the user who created it."""
    owner_tenant_str, _, owner_user_str = stored.partition(":")
    if uuid.UUID(owner_tenant_str) != tenant_id:
        return False
    if owner_user_str and uuid.UUID(owner_user_str) != user_id:
        return False
    return True
