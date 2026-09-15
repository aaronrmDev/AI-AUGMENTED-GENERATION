import pytest

from src.cag.domain.entities import ConversationTurn
from src.cag.domain.session_metrics import is_flat_per_turn, recompute_ratio
from src.cag.infrastructure.kvzip_session import KVzipSession
from src.cag.infrastructure.memserve_session import MemServeSession
from src.cag.infrastructure.naive_recompute_session import (
    IncrementalAppendSession,
    NaiveRecomputeSession,
)
from src.cag.infrastructure.rocketkv_mt_session import RocketKVMTSession
from src.cag.infrastructure.sglang_session import SGLangSession
from src.cag.infrastructure.shadowkv_compressor import ShadowKVCompressor
from src.cag.infrastructure.shadowkv_session import ShadowKVSession


def _linear_session(turns: int = 10, tokens_per_turn: int = 100) -> list[ConversationTurn]:
    return [
        ConversationTurn(
            turn_id=f"t{i}",
            new_tokens=tokens_per_turn,
            parent_turn_id=None if i == 0 else f"t{i - 1}",
        )
        for i in range(turns)
    ]


def _branching_session() -> list[ConversationTurn]:
    # One shared 300-token trunk, then four branches off its tip -- the
    # agent-program shape SGLang is built for.
    trunk = [
        ConversationTurn("t0", 100, None),
        ConversationTurn("t1", 100, "t0"),
        ConversationTurn("t2", 100, "t1"),
    ]
    branches = [ConversationTurn(f"b{i}", 50, "t2") for i in range(4)]
    return trunk + branches


def test_the_naive_baseline_grows_with_every_turn():
    run = NaiveRecomputeSession().run_session(_linear_session())
    assert is_flat_per_turn(run.per_turn_recompute) is False
    # Turn 10 costs ten times turn 1: the reprint-the-book shape.
    assert run.per_turn_recompute[-1] == 10 * run.per_turn_recompute[0]


def test_incremental_append_achieves_a_flat_per_turn_cost():
    # CAG.md's headline claim, as a structural assertion rather than a
    # smaller total: the cost per turn stops growing.
    run = IncrementalAppendSession().run_session(_linear_session())
    assert is_flat_per_turn(run.per_turn_recompute) is True
    naive = NaiveRecomputeSession().run_session(_linear_session())
    # 10 turns: 1000 tokens against the naive 5500.
    assert recompute_ratio(run.tokens_recomputed, naive.tokens_recomputed) == pytest.approx(
        1000 / 5500
    )


def test_rocketkv_mt_caps_the_current_turn_while_retaining_everything():
    turns = _linear_session(turns=6, tokens_per_turn=100)
    run = RocketKVMTSession(current_turn_budget=40).run_session(turns)
    # Selective about the present turn...
    assert all(cost == 40 for cost in run.per_turn_recompute)
    # ...but retains the full history for turns not yet asked.
    full = IncrementalAppendSession().run_session(turns)
    assert run.peak_tokens_held == full.peak_tokens_held


def test_rocketkv_mt_rejects_a_non_positive_budget():
    with pytest.raises(ValueError):
        RocketKVMTSession(current_turn_budget=0)


def test_kvzip_pays_its_overhead_once_so_cost_per_turn_falls_as_the_session_runs():
    short = KVzipSession(compression_ratio=0.5, compression_overhead_tokens=500).run_session(
        _linear_session(turns=2)
    )
    long = KVzipSession(compression_ratio=0.5, compression_overhead_tokens=500).run_session(
        _linear_session(turns=20)
    )
    short_per_turn = short.tokens_recomputed / 2
    long_per_turn = long.tokens_recomputed / 20
    # Amortisation: the same fixed overhead costs far less per query the
    # more queries eventually arrive, which is KVzip's whole argument.
    assert long_per_turn < short_per_turn


def test_kvzip_holds_a_compressed_footprint():
    turns = _linear_session(turns=8)
    kvzip = KVzipSession(compression_ratio=0.5, compression_overhead_tokens=100).run_session(turns)
    uncompressed = IncrementalAppendSession().run_session(turns)
    assert kvzip.peak_tokens_held < uncompressed.peak_tokens_held


def test_shadowkv_reuses_the_subspace_of_the_sequence_it_continues():
    turns = _linear_session(turns=10)
    session = ShadowKVSession(
        compressor=ShadowKVCompressor(rank=8, sparsity_ratio=0.2, bits=6),
        subspace_reuse=0.75,
    )
    run = session.run_session(turns)
    plain = IncrementalAppendSession().run_session(turns)
    # A continuation is not rebuilding its representation from scratch.
    assert run.tokens_recomputed < plain.tokens_recomputed
    assert run.per_turn_recompute[0] > run.per_turn_recompute[1]


def test_shadowkv_rejects_a_reuse_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError):
        ShadowKVSession(
            compressor=ShadowKVCompressor(rank=8, sparsity_ratio=0.2, bits=6),
            subspace_reuse=1.5,
        )


def test_memserve_is_the_only_strategy_that_moves_history_across_a_network():
    turns = _linear_session(turns=8)
    memserve = MemServeSession(local_resident_tokens=200).run_session(turns)
    local = IncrementalAppendSession().run_session(turns)
    assert memserve.transfer_tokens > 0
    assert local.transfer_tokens == 0
    # Elastic pool: local residency is capped where a single-machine
    # strategy's peak is not.
    assert memserve.peak_tokens_held < local.peak_tokens_held


def test_memserve_transfers_nothing_when_the_whole_session_fits_locally():
    run = MemServeSession(local_resident_tokens=100_000).run_session(_linear_session())
    assert run.transfer_tokens == 0


def test_sglang_materialises_a_shared_trunk_once_across_branches():
    branching = _branching_session()
    sglang = SGLangSession().run_session(branching)
    naive = NaiveRecomputeSession().run_session(branching)
    assert sglang.tokens_recomputed < naive.tokens_recomputed
    # Each of the four branches costs only its own 50 tokens, because
    # the 300-token trunk was already materialised by the turn before.
    assert sglang.per_turn_recompute[-4:] == [50, 50, 50, 50]


def test_sglang_degrades_to_incremental_append_on_a_flat_conversation():
    # CAG.md says plainly this suits agent workflows "rather than a
    # single flat conversation" -- so matching, not beating, plain
    # incremental append here is the correct result, not a shortfall.
    flat = _linear_session()
    assert (
        SGLangSession().run_session(flat).tokens_recomputed
        == IncrementalAppendSession().run_session(flat).tokens_recomputed
    )
