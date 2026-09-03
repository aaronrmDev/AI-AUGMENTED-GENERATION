import uuid

_NAMESPACE = uuid.NAMESPACE_OID


def cache_key(user_id: uuid.UUID, mag_content_key: str) -> uuid.UUID:
    """The FrozenCache document_id for one user's promoted MAG content.

    FrozenCache (Batch D) has no user_id parameter at all -- it's a
    tenant-wide shared cache by design. Promoting inherently personal MAG
    content into it without namespacing the key would let a different
    user's identical query read this user's promoted content back out of
    the shared cache. Folding user_id into the key itself keeps two
    users' entries for "the same" conceptual mag_content_key from ever
    colliding, without adding a parameter FrozenCache's own contract
    doesn't have.
    """
    return uuid.uuid5(_NAMESPACE, f"{user_id}:{mag_content_key}")


def tracker_key(mag_content_key: str) -> uuid.UUID:
    """The UserScopedAccessFrequencyTracker document_id for one piece of
    MAG content.

    Unlike FrozenCache, UserScopedAccessFrequencyTracker (Batch E) already
    takes a real user_id argument of its own -- folding user_id into this
    key too would double-scope it redundantly, and would also make this
    key differ from cache_key's for the same conceptual content, which
    would just be confusing. This key is content-only; the caller passes
    user_id to the tracker directly, exactly as its own port expects.
    """
    return uuid.uuid5(_NAMESPACE, mag_content_key)
