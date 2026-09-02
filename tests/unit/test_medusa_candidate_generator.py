import torch

from src.cag.infrastructure.medusa_candidate_generator import MedusaCandidateGenerator


class _FakeMultiLayerCausalLM(torch.nn.Module):
    # A tiny, deterministic stand-in for a real transformers causal LM --
    # no download, no real weights. Every (layer, position) pair gets its
    # OWN distinct one-hot hidden vector, and lm_head is an identity-like
    # matrix mapping each one-hot vector to its own uniquely identifiable
    # predicted token (token id = layer_index * num_positions +
    # position_index). This makes a wrong-layer or wrong-position
    # indexing bug in MedusaCandidateGenerator.propose's own
    # hidden_states[-1][0, -1] selection directly observable as a
    # different predicted token -- review-caught: an earlier version of
    # this fake broadcast the identical hidden vector across every
    # position and exposed only one layer, so a mutation to
    # hidden_states[0][0, 0] (first layer, first position instead of
    # last layer, last position) passed every test in this file
    # unchanged. Confirmed this fake actually closes that gap by
    # reproducing the exact mutation against it before finalizing.
    def __init__(self, num_layers: int = 2, num_positions: int = 3) -> None:
        super().__init__()
        self._num_layers = num_layers
        self._num_positions = num_positions
        hidden_size = num_layers * num_positions
        self._lm_head = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self._lm_head.weight.copy_(torch.eye(hidden_size) * 10.0)

    def get_output_embeddings(self) -> torch.nn.Module:
        return self._lm_head

    def token_at(self, layer: int, position: int) -> int:
        # The token id this fake predicts if (and only if) exactly this
        # (layer, position) pair's hidden vector is what gets selected.
        return layer * self._num_positions + position

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False) -> object:
        hidden_size = self._num_layers * self._num_positions
        layers = []
        for layer in range(self._num_layers):
            layer_hidden = torch.zeros(1, self._num_positions, hidden_size)
            for position in range(self._num_positions):
                layer_hidden[0, position, self.token_at(layer, position)] = 1.0
            layers.append(layer_hidden)
        return type("FakeOutput", (), {"hidden_states": tuple(layers)})()


def test_reads_the_last_layers_last_position_not_any_other():
    # The exact indexing contract MedusaCandidateGenerator.propose relies
    # on: hidden_states[-1][0, -1] -- last layer, last (most recent)
    # position. Every other (layer, position) pair predicts a distinct,
    # wrong token, so this pins the real selection down precisely rather
    # than merely being consistent with several possible selections.
    model = _FakeMultiLayerCausalLM(num_layers=2, num_positions=3)
    generator = MedusaCandidateGenerator(model, num_heads=1)

    candidates = generator.propose(prompt_tokens=[1, 2, 3], generated_tokens=[], num_candidates=1)

    assert candidates == [model.token_at(layer=1, position=2)]
    # Sanity: every OTHER (layer, position) combination predicts a
    # genuinely different token, confirming the fixture can actually
    # distinguish a wrong selection rather than all pairs coincidentally
    # aliasing to the same value.
    other_tokens = {
        model.token_at(layer, position)
        for layer in range(2)
        for position in range(3)
        if (layer, position) != (1, 2)
    }
    assert candidates[0] not in other_tokens


def test_every_head_produces_the_identical_prediction_when_untrained():
    # Disclosed, expected consequence of the warm-start-by-copying
    # simplification: every head is an IDENTICAL copy of lm_head applied
    # to the SAME hidden state, so every head predicts the SAME token --
    # heads 1+ have no way to know they're supposed to predict a
    # different, later position, since nothing in copying weights
    # teaches that. This is the real mechanism the design spec's own
    # disclosure describes, pinned down deterministically here rather
    # than only observed once against a real, slower model.
    model = _FakeMultiLayerCausalLM(num_layers=1, num_positions=2)
    generator = MedusaCandidateGenerator(model, num_heads=4)

    candidates = generator.propose(prompt_tokens=[1, 2], generated_tokens=[], num_candidates=4)

    expected = model.token_at(layer=0, position=1)
    assert candidates == [expected, expected, expected, expected]


def test_proposes_at_most_num_candidates_even_with_more_heads_available():
    model = _FakeMultiLayerCausalLM(num_layers=1, num_positions=2)
    generator = MedusaCandidateGenerator(model, num_heads=4)

    candidates = generator.propose(prompt_tokens=[1, 2], generated_tokens=[], num_candidates=2)

    assert len(candidates) == 2


def test_returns_empty_for_an_empty_prompt_and_no_generated_tokens():
    model = _FakeMultiLayerCausalLM(num_layers=1, num_positions=2)
    generator = MedusaCandidateGenerator(model, num_heads=2)

    assert generator.propose(prompt_tokens=[], generated_tokens=[], num_candidates=2) == []
