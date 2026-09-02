import pytest

from src.cag.application.speculative_decode import SpeculativeDecode
from src.cag.domain.entities import VerificationResult
from tests.unit.cag_fakes import FakeCandidateGenerator, FakeTargetModel


def test_full_acceptance_runs_one_forward_pass_per_batch_of_candidates():
    # 4 candidates proposed, all 4 accepted plus a bonus token -- 5 new
    # tokens from ONE forward pass, exactly the O(N/K) saving speculative
    # decoding exists for.
    candidate_generator = FakeCandidateGenerator([[10, 11, 12, 13]])
    target_model = FakeTargetModel(
        [VerificationResult(accepted_tokens=[10, 11, 12, 13], bonus_token=14)]
    )
    use_case = SpeculativeDecode()

    run = use_case.execute(
        target_model=target_model,
        candidate_generator=candidate_generator,
        prompt_tokens=[1, 2, 3],
        max_new_tokens=5,
        num_candidates=4,
    )

    assert run.generated_tokens == [10, 11, 12, 13, 14]
    assert run.forward_passes == 1
    assert run.tokens_accepted_from_candidates == 4
    assert run.tokens_proposed == 4


def test_zero_acceptance_falls_back_to_one_token_per_forward_pass():
    # No candidates match (propose() returns []) -- the correctness floor:
    # speculative decoding must never do WORSE than naive one-token-per-
    # pass generation, only ever the same or better.
    candidate_generator = FakeCandidateGenerator([[], [], []])
    target_model = FakeTargetModel(
        [
            VerificationResult(accepted_tokens=[], bonus_token=20),
            VerificationResult(accepted_tokens=[], bonus_token=21),
            VerificationResult(accepted_tokens=[], bonus_token=22),
        ]
    )
    use_case = SpeculativeDecode()

    run = use_case.execute(
        target_model=target_model,
        candidate_generator=candidate_generator,
        prompt_tokens=[1, 2],
        max_new_tokens=3,
        num_candidates=4,
    )

    assert run.generated_tokens == [20, 21, 22]
    assert run.forward_passes == 3
    assert run.tokens_accepted_from_candidates == 0
    assert run.tokens_proposed == 0


def test_partial_acceptance_retries_from_the_first_mismatch():
    # 4 candidates proposed, only the first 2 accepted (position 2
    # mismatched) -- the bonus token replaces the rejected candidate at
    # the mismatch point, and the next round proposes fresh candidates
    # from the now-longer generated tail.
    candidate_generator = FakeCandidateGenerator([[10, 11, 99, 98], [30, 31]])
    target_model = FakeTargetModel(
        [
            VerificationResult(accepted_tokens=[10, 11], bonus_token=20),
            VerificationResult(accepted_tokens=[30], bonus_token=40),
        ]
    )
    use_case = SpeculativeDecode()

    run = use_case.execute(
        target_model=target_model,
        candidate_generator=candidate_generator,
        prompt_tokens=[1],
        max_new_tokens=4,
        num_candidates=4,
    )

    assert run.generated_tokens == [10, 11, 20, 30]
    assert run.forward_passes == 2
    assert run.tokens_accepted_from_candidates == 3
    assert run.tokens_proposed == 6
    # Second round's candidate generator call sees the tokens generated
    # by the first round's accept+bonus, not a stale/empty tail.
    assert candidate_generator.calls[1][1] == [10, 11, 20]


def test_stops_exactly_at_max_new_tokens_even_mid_candidate_batch():
    # Verification accepts more tokens than are actually needed to reach
    # max_new_tokens -- the run must truncate rather than overshoot, and
    # must not append a bonus token past the requested length either.
    candidate_generator = FakeCandidateGenerator([[10, 11, 12, 13]])
    target_model = FakeTargetModel(
        [VerificationResult(accepted_tokens=[10, 11, 12, 13], bonus_token=14)]
    )
    use_case = SpeculativeDecode()

    run = use_case.execute(
        target_model=target_model,
        candidate_generator=candidate_generator,
        prompt_tokens=[1],
        max_new_tokens=2,
        num_candidates=4,
    )

    assert run.generated_tokens == [10, 11]
    assert run.forward_passes == 1
    # Review-caught: an earlier version counted all 4 verified-accepted
    # candidates here, even though 2 of them were discarded by
    # truncation -- tokens_accepted_from_candidates must only count
    # tokens that actually survived into generated_tokens, or the
    # acceptance rate this field exists to compute (see
    # SpeculativeDecodingRun's own docstring) silently overstates itself
    # whenever a batch's accepted tokens overshoot the remaining budget.
    assert run.tokens_accepted_from_candidates == 2
    assert run.tokens_proposed == 4


def test_target_model_receives_the_full_context_so_far():
    # verify_candidates must see prompt + everything generated so far,
    # not just the prompt or just the latest candidates -- otherwise the
    # target model has no way to know what it's continuing from.
    candidate_generator = FakeCandidateGenerator([[50], [60]])
    target_model = FakeTargetModel(
        [
            VerificationResult(accepted_tokens=[50], bonus_token=51),
            VerificationResult(accepted_tokens=[60], bonus_token=61),
        ]
    )
    use_case = SpeculativeDecode()

    use_case.execute(
        target_model=target_model,
        candidate_generator=candidate_generator,
        prompt_tokens=[1, 2, 3],
        max_new_tokens=4,
        num_candidates=1,
    )

    assert target_model.calls[0][0] == [1, 2, 3]
    assert target_model.calls[1][0] == [1, 2, 3, 50, 51]
    # Review-caught: FakeTargetModel's scripted responses don't depend on
    # what candidates it's actually given, so nothing previously proved
    # propose()'s real return value reaches verify_candidates() intact --
    # a regression dropping, reordering, or stale-caching the candidates
    # passed between them would have shipped undetected.
    assert target_model.calls[0][1] == [50]
    assert target_model.calls[1][1] == [60]


def test_raises_if_verification_makes_no_progress_under_budget():
    # Review-caught: a TargetModel/CandidateGenerator pairing that
    # violates VerificationResult's own documented contract ("bonus_token
    # is None only when max_new_tokens was already reached") -- returning
    # no accepted tokens AND no bonus token while still under budget --
    # would otherwise make the loop spin forever, silently incrementing
    # forward_passes with nothing to show for it. Every shipped
    # TargetModel (HFTargetModel) always returns a real bonus_token
    # unless the budget is already exhausted, so this never fires in
    # practice today -- this is a defensive guard against a future
    # contract violation, not a currently-reachable bug.
    candidate_generator = FakeCandidateGenerator([[]])
    target_model = FakeTargetModel([VerificationResult(accepted_tokens=[], bonus_token=None)])
    use_case = SpeculativeDecode()

    with pytest.raises(RuntimeError, match="no progress"):
        use_case.execute(
            target_model=target_model,
            candidate_generator=candidate_generator,
            prompt_tokens=[1],
            max_new_tokens=3,
            num_candidates=2,
        )
