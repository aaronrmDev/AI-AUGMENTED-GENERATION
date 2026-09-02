import torch

from src.cag.domain.entities import VerificationResult
from src.cag.domain.ports import TargetModel


class HFTargetModel(TargetModel):
    # One real forward pass over (tokens + candidates). Position j's
    # logits are conditioned on tokens[0:j] and predict what comes AFTER
    # position j -- so the logits that predict candidates[i] sit at
    # index len(tokens) - 1 + i, not at len(tokens) + i. Verified live
    # against a real distilgpt2 forward pass before writing this: a
    # genuinely correct candidate sequence (the model's own real greedy
    # continuation, computed independently one token at a time) gets
    # fully accepted through this exact indexing, and the bonus token
    # (argmax at the very last logits position) exactly matches what a
    # real one-more-step continuation produces.
    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self._model = model
        self._device = device

    def verify_candidates(self, tokens: list[int], candidates: list[int]) -> VerificationResult:
        full_sequence = tokens + candidates
        input_ids = torch.tensor([full_sequence], device=self._device)
        with torch.no_grad():
            logits = self._model(input_ids).logits[0]

        accepted: list[int] = []
        for i, candidate in enumerate(candidates):
            predicted = int(torch.argmax(logits[len(tokens) - 1 + i]).item())
            if predicted != candidate:
                # Mismatch: the bonus token IS the target's own real
                # prediction at this exact position -- no extra work,
                # it's already been computed as `predicted` above.
                return VerificationResult(accepted_tokens=accepted, bonus_token=predicted)
            accepted.append(candidate)

        # Every candidate matched -- the bonus is the target's own
        # greedy continuation past the full accepted sequence, read
        # from the same forward pass's very last logits position.
        bonus = int(torch.argmax(logits[-1]).item())
        return VerificationResult(accepted_tokens=accepted, bonus_token=bonus)
