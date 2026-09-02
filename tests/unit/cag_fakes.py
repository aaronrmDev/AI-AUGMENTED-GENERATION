from src.cag.domain.entities import VerificationResult
from src.cag.domain.ports import CandidateGenerator, TargetModel


class FakeCandidateGenerator(CandidateGenerator):
    # Returns a scripted sequence of candidate lists, one per call --
    # lets a test control exactly what's proposed on each loop iteration
    # without needing a real n-gram match to actually occur.
    def __init__(self, scripted_candidates: list[list[int]]) -> None:
        self._scripted = iter(scripted_candidates)
        self.calls: list[tuple[list[int], list[int], int]] = []

    def propose(
        self, prompt_tokens: list[int], generated_tokens: list[int], num_candidates: int
    ) -> list[int]:
        self.calls.append((list(prompt_tokens), list(generated_tokens), num_candidates))
        return next(self._scripted, [])


class FakeTargetModel(TargetModel):
    # Returns a scripted sequence of VerificationResults, one per call --
    # a real target model's accept/reject/bonus decision is exactly what
    # SpeculativeDecode's own loop logic needs to be tested against,
    # independent of any real model's actual predictions.
    def __init__(self, scripted_results: list[VerificationResult]) -> None:
        self._scripted = iter(scripted_results)
        self.calls: list[tuple[list[int], list[int]]] = []

    def verify_candidates(self, tokens: list[int], candidates: list[int]) -> VerificationResult:
        self.calls.append((list(tokens), list(candidates)))
        return next(self._scripted)
