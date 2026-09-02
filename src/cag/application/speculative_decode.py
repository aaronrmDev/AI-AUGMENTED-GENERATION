from src.cag.domain.entities import SpeculativeDecodingRun
from src.cag.domain.ports import CandidateGenerator, TargetModel


class SpeculativeDecode:
    # The one loop shared by Medusa, Lookahead Decoding, and Prompt
    # Lookup Decoding -- they differ only in what CandidateGenerator
    # they plug in. propose -> verify (one real forward pass) -> accept
    # the prefix up to the first mismatch plus its free bonus token ->
    # repeat. When propose() returns no candidates, verify_candidates
    # still returns exactly one bonus token per call -- the correctness
    # floor: this loop never does worse than naive one-token-per-forward-
    # pass generation, only ever the same or better.
    def execute(
        self,
        target_model: TargetModel,
        candidate_generator: CandidateGenerator,
        prompt_tokens: list[int],
        max_new_tokens: int,
        num_candidates: int,
    ) -> SpeculativeDecodingRun:
        generated_tokens: list[int] = []
        forward_passes = 0
        tokens_accepted_from_candidates = 0
        tokens_proposed = 0

        while len(generated_tokens) < max_new_tokens:
            candidates = candidate_generator.propose(
                prompt_tokens, generated_tokens, num_candidates
            )
            tokens_proposed += len(candidates)

            result = target_model.verify_candidates(prompt_tokens + generated_tokens, candidates)
            forward_passes += 1
            tokens_accepted_from_candidates += len(result.accepted_tokens)

            generated_tokens.extend(result.accepted_tokens)
            if result.bonus_token is not None and len(generated_tokens) < max_new_tokens:
                generated_tokens.append(result.bonus_token)
            if len(generated_tokens) > max_new_tokens:
                generated_tokens = generated_tokens[:max_new_tokens]

        return SpeculativeDecodingRun(
            generated_tokens=generated_tokens,
            forward_passes=forward_passes,
            tokens_accepted_from_candidates=tokens_accepted_from_candidates,
            tokens_proposed=tokens_proposed,
        )
