import torch

from src.cag.infrastructure.medusa_candidate_generator import MedusaCandidateGenerator


class _FakeCausalLMWithHiddenState(torch.nn.Module):
    # A tiny, deterministic stand-in for a real transformers causal LM --
    # no download, no real weights. get_output_embeddings() returns a
    # real, tiny nn.Linear with hand-set weights so the exact predicted
    # token from applying it to a known hidden state is fully
    # controllable, and forward(..., output_hidden_states=True) returns
    # that same known hidden state as its final layer's last position.
    def __init__(self) -> None:
        super().__init__()
        # hidden_size=2, vocab_size=3. Weight rows are one-hot-ish so a
        # hidden state of [1.0, 0.0] dot-products highest against row 0
        # (predicts token 0), and [0.0, 1.0] against row 1 (token 1).
        self._lm_head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self._lm_head.weight.copy_(
                torch.tensor([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
            )

    def get_output_embeddings(self) -> torch.nn.Module:
        return self._lm_head

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False) -> object:
        num_positions = input_ids.shape[1]
        # A hidden state of [1.0, 0.0] at every position -- applying the
        # hand-set lm_head above to this always predicts token 0.
        hidden = torch.tensor([[1.0, 0.0]] * num_positions).unsqueeze(0)
        return type("FakeOutput", (), {"hidden_states": (hidden,)})()


def test_head_zero_reproduces_the_real_next_token_prediction():
    # The mathematical guarantee the design spec relies on: head 0 is an
    # exact copy of lm_head, so its prediction on the real final hidden
    # state is identical to a genuine next-token prediction -- here,
    # token 0, matching the hand-set weights' own top row.
    model = _FakeCausalLMWithHiddenState()
    generator = MedusaCandidateGenerator(model, num_heads=3)

    candidates = generator.propose(prompt_tokens=[1, 2], generated_tokens=[], num_candidates=1)

    assert candidates == [0]


def test_every_head_produces_the_identical_prediction_when_untrained():
    # Disclosed, expected consequence of the warm-start-by-copying
    # simplification: every head is an IDENTICAL copy of lm_head applied
    # to the SAME hidden state, so every head predicts the SAME token --
    # heads 1+ have no way to know they're supposed to predict a
    # different, later position, since nothing in copying weights
    # teaches that. This is the real mechanism the design spec's own
    # disclosure describes, pinned down deterministically here rather
    # than only observed once against a real, slower model.
    model = _FakeCausalLMWithHiddenState()
    generator = MedusaCandidateGenerator(model, num_heads=4)

    candidates = generator.propose(prompt_tokens=[1, 2], generated_tokens=[], num_candidates=4)

    assert candidates == [0, 0, 0, 0]


def test_proposes_at_most_num_candidates_even_with_more_heads_available():
    model = _FakeCausalLMWithHiddenState()
    generator = MedusaCandidateGenerator(model, num_heads=4)

    candidates = generator.propose(prompt_tokens=[1, 2], generated_tokens=[], num_candidates=2)

    assert len(candidates) == 2


def test_returns_empty_for_an_empty_prompt_and_no_generated_tokens():
    model = _FakeCausalLMWithHiddenState()
    generator = MedusaCandidateGenerator(model, num_heads=2)

    assert generator.propose(prompt_tokens=[], generated_tokens=[], num_candidates=2) == []
