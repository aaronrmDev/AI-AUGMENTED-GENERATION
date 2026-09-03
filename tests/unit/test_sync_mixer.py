import uuid

from src.orchestration.domain.sync_mixer import content_hash, reconcile


def test_no_cached_entry_means_nothing_to_reconcile():
    result = reconcile(
        cached_content_hash=None, authoritative_content="anything", document_id=uuid.uuid4()
    )
    assert result is None


def test_matching_content_is_already_in_sync():
    content = "return policy: 30 days"

    result = reconcile(
        cached_content_hash=content_hash(content),
        authoritative_content=content,
        document_id=uuid.uuid4(),
    )

    assert result is None


def test_mismatched_content_is_a_conflict_and_rag_wins():
    stale_content = "price: $100"
    fresh_content = "price: $80"
    document_id = uuid.uuid4()

    result = reconcile(
        cached_content_hash=content_hash(stale_content),
        authoritative_content=fresh_content,
        document_id=document_id,
    )

    assert result is not None
    assert result.document_id == document_id
    assert result.cached_content_hash == content_hash(stale_content)
    assert result.authoritative_content_hash == content_hash(fresh_content)
    assert result.cached_content_hash != result.authoritative_content_hash


def test_content_hash_is_deterministic_and_content_sensitive():
    assert content_hash("a") == content_hash("a")
    assert content_hash("a") != content_hash("b")
