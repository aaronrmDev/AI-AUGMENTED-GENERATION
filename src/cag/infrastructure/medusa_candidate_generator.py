import copy
from typing import cast

import torch

from src.cag.domain.ports import CandidateGenerator


class MedusaCandidateGenerator(CandidateGenerator):
    # CAG.md: "attaches multiple draft heads directly onto the target
    # model instead of using a wholly separate draft model." Real
    # architecture: num_heads extra Linear(hidden_size, vocab_size)
    # layers, one per future position (head 0 predicts the immediate
    # next token, head 1 the one after, etc.), all applied to the SAME
    # final hidden state from ONE forward pass -- no autoregressive
    # drafting, which is the whole point of attaching heads to the
    # target model itself rather than running a second model.
    #
    # Disclosed simplification (per the design spec): real Medusa heads
    # are fine-tuned on a training run this batch doesn't do. These
    # heads are warm-started as exact copies of the target model's own
    # lm_head, not left randomly initialized. That warm start makes
    # head 0 mathematically identical to a real next-token prediction
    # (accepted essentially always), but since every head is an
    # IDENTICAL copy applied to the SAME hidden state, every head
    # produces the SAME top prediction -- heads 1+ have no way to know
    # they're supposed to predict a DIFFERENT, later position, since
    # nothing in copying weights teaches that; only real training with a
    # future-offset-aware loss does. This is an honest, expected
    # consequence of the disclosed simplification, not a bug -- it's
    # exactly why real Medusa needs the training this batch doesn't
    # attempt, and the measured acceptance rate beyond the first
    # candidate is reported honestly in the batch's own evaluation
    # report, not hidden.
    def __init__(self, model: torch.nn.Module, num_heads: int, device: str = "cpu") -> None:
        self._model = model
        self._device = device
        # torch.nn.Module has no static declaration of get_output_embeddings
        # (a transformers PreTrainedModel API, and transformers ships no
        # inline type stubs) -- Module's own __getattr__ fallback types an
        # unknown attribute as Tensor | Module, which mypy correctly refuses
        # to call directly (a Tensor isn't callable). Every real HF causal
        # LM's get_output_embeddings() actually returns a real nn.Module (a
        # Linear layer), never a bare Tensor.
        lm_head = cast(torch.nn.Module, model.get_output_embeddings())  # type: ignore[operator]
        self._heads: list[torch.nn.Module] = []
        for _ in range(num_heads):
            head = copy.deepcopy(lm_head)
            head.to(device)
            self._heads.append(head)

    def propose(
        self, prompt_tokens: list[int], generated_tokens: list[int], num_candidates: int
    ) -> list[int]:
        context = prompt_tokens + generated_tokens
        if not context:
            return []
        input_ids = torch.tensor([context], device=self._device)
        with torch.no_grad():
            hidden_state = self._model(input_ids, output_hidden_states=True).hidden_states[-1][
                0, -1
            ]
            candidates = [
                int(torch.argmax(head(hidden_state)).item())
                for head in self._heads[: min(num_candidates, len(self._heads))]
            ]
        return candidates
