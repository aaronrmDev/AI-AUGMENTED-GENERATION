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
            length_before_this_round = len(generated_tokens)
            candidates = candidate_generator.propose(
                prompt_tokens, generated_tokens, num_candidates
            )
            tokens_proposed += len(candidates)

            result = target_model.verify_candidates(prompt_tokens + generated_tokens, candidates)
            forward_passes += 1

            # Cap what gets counted (and appended) at the remaining
            # budget BEFORE counting -- review-caught: incrementing
            # tokens_accepted_from_candidates by the full verified count
            # and only truncating generated_tokens afterward meant the
            # counter could claim more "accepted" tokens than actually
            # survived into the output, silently inflating the acceptance
            # rate this run's own stats exist to report accurately.
            remaining_budget = max_new_tokens - len(generated_tokens)
            accepted_within_budget = result.accepted_tokens[:remaining_budget]
            tokens_accepted_from_candidates += len(accepted_within_budget)
            generated_tokens.extend(accepted_within_budget)
            if result.bonus_token is not None and len(generated_tokens) < max_new_tokens:
                generated_tokens.append(result.bonus_token)

            # Forward-progress guard (review-caught, defensive): every
            # shipped TargetModel returns a real bonus_token unless
            # max_new_tokens was already reached, so this never fires
            # today -- but nothing else in this loop stops a future
            # TargetModel/CandidateGenerator pairing that violated that
            # contract (accepted_tokens=[] and bonus_token=None while
            # under budget) from spinning forever, silently incrementing
            # forward_passes with no error. Fail loudly instead.
            if len(generated_tokens) == length_before_this_round:
                raise RuntimeError(
                    "speculative decoding made no progress this round -- "
                    "TargetModel.verify_candidates returned no accepted tokens and no "
                    "bonus_token while still under max_new_tokens, violating its own contract"
                )

        return SpeculativeDecodingRun(
            generated_tokens=generated_tokens,
            forward_passes=forward_passes,
            tokens_accepted_from_candidates=tokens_accepted_from_candidates,
            tokens_proposed=tokens_proposed,
        )
