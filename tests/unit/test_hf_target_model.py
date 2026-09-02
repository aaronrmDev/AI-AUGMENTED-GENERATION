import torch

from src.cag.infrastructure.hf_target_model import HFTargetModel


class _FakeLogitsModel(torch.nn.Module):
    # A tiny, deterministic stand-in for a real transformers causal LM --
    # no download, no real weights, an exact and fully controllable
    # argmax at every position, so HFTargetModel's own indexing
    # arithmetic (logits[len(tokens) - 1 + i] predicting candidates[i])
    # can be pinned down by a fast, deterministic unit test instead of
    # only ever exercised against a real, slower distilgpt2 forward pass
    # (which the integration test still covers separately for real-model
    # fidelity).
    def __init__(self, next_token_by_position: list[int], vocab_size: int = 1500) -> None:
        super().__init__()
        self._next_token_by_position = next_token_by_position
        self._vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> object:
        num_positions = input_ids.shape[1]
        logits = torch.full((1, num_positions, self._vocab_size), -1e9)
        for position in range(num_positions):
            logits[0, position, self._next_token_by_position[position]] = 1e9
        return type("FakeOutput", (), {"logits": logits})()


def test_accepts_all_candidates_and_returns_the_free_bonus_token():
    tokens = [1, 2, 3]
    candidates = [10, 11, 12]
    # Position len(tokens)-1+i predicts candidates[i]; position 5 (the
    # last, len(tokens)-1+len(candidates)) is the free bonus prediction
    # once every candidate has matched. Positions 0-1 are never consulted
    # by verify_candidates (only tokens[len(tokens)-1:] onward matter).
    next_token_by_position = [0, 0, 10, 11, 12, 42]
    model = _FakeLogitsModel(next_token_by_position)

    result = HFTargetModel(model).verify_candidates(tokens, candidates)

    assert result.accepted_tokens == [10, 11, 12]
    assert result.bonus_token == 42


def test_rejects_from_the_first_mismatch_and_the_bonus_is_the_targets_own_prediction():
    tokens = [1, 2, 3]
    candidates = [10, 11, 12]
    # Position 2 predicts 10 (matches candidates[0]) -> accept. Position 3
    # predicts 11 (matches candidates[1]) -> accept. Position 4 predicts
    # 99, NOT 12 (candidates[2]) -> reject here; the bonus is exactly
    # that mismatch prediction (99), not candidates[2] and not
    # position 5's prediction (which is never even consulted, since
    # verification stops at the first mismatch).
    next_token_by_position = [0, 0, 10, 11, 99, 77]
    model = _FakeLogitsModel(next_token_by_position)

    result = HFTargetModel(model).verify_candidates(tokens, candidates)

    assert result.accepted_tokens == [10, 11]
    assert result.bonus_token == 99


def test_rejects_immediately_when_the_first_candidate_mismatches():
    tokens = [1, 2, 3]
    candidates = [10, 11, 12]
    next_token_by_position = [0, 0, 999, 11, 12, 42]
    model = _FakeLogitsModel(next_token_by_position)

    result = HFTargetModel(model).verify_candidates(tokens, candidates)

    assert result.accepted_tokens == []
    assert result.bonus_token == 999


def test_zero_candidates_still_returns_exactly_one_bonus_token():
    # The correctness floor SpeculativeDecode's own loop depends on:
    # verifying an empty candidate list still runs one real forward pass
    # and returns one real token, matching naive one-token-per-pass
    # generation exactly.
    tokens = [1, 2, 3]
    next_token_by_position = [0, 0, 55]
    model = _FakeLogitsModel(next_token_by_position)

    result = HFTargetModel(model).verify_candidates(tokens, [])

    assert result.accepted_tokens == []
    assert result.bonus_token == 55
