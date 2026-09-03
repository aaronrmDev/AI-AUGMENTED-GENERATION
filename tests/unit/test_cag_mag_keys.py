import uuid

from src.orchestration.domain.cag_mag_keys import cache_key, tracker_key

_USER_A = uuid.uuid4()
_USER_B = uuid.uuid4()


def test_cache_key_is_deterministic():
    assert cache_key(_USER_A, "preferred_library") == cache_key(_USER_A, "preferred_library")


def test_cache_key_differs_across_users_for_the_same_content_key():
    assert cache_key(_USER_A, "preferred_library") != cache_key(_USER_B, "preferred_library")


def test_cache_key_differs_across_content_keys_for_the_same_user():
    assert cache_key(_USER_A, "preferred_library") != cache_key(_USER_A, "other_key")


def test_tracker_key_is_deterministic():
    assert tracker_key("preferred_library") == tracker_key("preferred_library")


def test_tracker_key_does_not_vary_by_user():
    # Deliberately content-only -- UserScopedAccessFrequencyTracker already
    # takes a real user_id argument of its own, so this key must not also
    # encode it (that would double-scope and desync it from cache_key).
    assert tracker_key("preferred_library") == tracker_key("preferred_library")


def test_tracker_key_and_cache_key_are_unrelated_derivations():
    # Not required to differ or match for any particular user/content
    # pair -- just documenting that they're two independent derivations,
    # not aliases of each other.
    assert tracker_key("preferred_library") != cache_key(_USER_A, "preferred_library")
