from src.mag.infrastructure._graph_write_safety import best_effort_graph_write


async def test_best_effort_graph_write_swallows_an_exception_from_the_write():
    # This is the core contract CaptureEpisode/RecordSemanticFact/
    # ConsolidateEpisodes all depend on -- a review found it had never
    # actually been exercised by any test in this batch (every unit test's
    # FakeMemoryGraphRepository never raises, and every integration test's
    # real Neo4j is healthy), so a regression that removed or narrowed the
    # try/except would have passed the whole suite unnoticed.
    ran = False

    async def _raise() -> None:
        nonlocal ran
        # Set before raising, not after -- a re-review caught that the
        # original version of this test only proved "no exception
        # propagated," which best_effort_graph_write silently discarding
        # the coroutine WITHOUT ever awaiting it would satisfy just as
        # well as actually catching a real exception. This flag is the
        # only way to tell "the write ran and its exception was caught"
        # apart from "the write never ran at all."
        ran = True
        raise RuntimeError("neo4j connection refused")

    await best_effort_graph_write(_raise(), "test write")

    assert ran is True  # the coroutine actually executed and raised
    # No exception propagated past the call above -- if it had, this test
    # itself would have failed before reaching this line.


async def test_best_effort_graph_write_lets_a_successful_write_complete_normally():
    called = False

    async def _write() -> None:
        nonlocal called
        called = True

    await best_effort_graph_write(_write(), "test write")

    assert called is True


async def test_best_effort_graph_write_swallows_a_type_error_too():
    # Not just infra-flavored exceptions -- best_effort_graph_write's own
    # docstring commits to catching bare Exception broadly (a deliberate
    # trade-off, since there's no principled way to distinguish "a genuine
    # driver failure" from "a caller passed the wrong type" without an
    # exhaustive, speculative allowlist of every neo4j exception class).
    ran = False

    async def _raise() -> None:
        nonlocal ran
        ran = True
        raise TypeError("bad argument")

    await best_effort_graph_write(_raise(), "test write")

    assert ran is True
