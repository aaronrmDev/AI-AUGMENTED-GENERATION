"""Deterministic success checks for the orchestration meta-layer comparison, one per
question in evaluation/scenarios/orchestration-meta-layer/queries.yaml, in file order.
Lives outside that directory because a hyphenated package can't be imported.

The three personal questions (8-10) also reject non-answers. Their facts' telltale
words appear in retrieved context that doesn't answer them: "sizes 9 and 10" in the
stock notice, "expedited" in the shipping policy. So an answer saying the context
doesn't specify the user's size still contains a 10. The earlier checks counted
exactly that as a success in the first comparison runs."""
import re
from collections.abc import Callable

_TEN = re.compile(r"\b(?:10|ten)\b")
_NON_ANSWER = re.compile(
    r"\b(?:does not|doesn't|do not|don't|did not|didn't)\s+"
    r"(?:contain|specify|mention|include|say|state|provide|indicate)\b"
    r"|\bnot\s+(?:specified|mentioned|provided|stated|known)\b"
    r"|\bno information\b"
    r"|\b(?:cannot|can't|unable to)\s+(?:determine|tell|confirm)\b"
)


def _normalize(answer: str) -> str:
    return answer.lower().replace("’", "'")


def is_non_answer(answer: str) -> bool:
    """Whether the answer says the context doesn't hold what was asked."""
    return _NON_ANSWER.search(_normalize(answer)) is not None


def _contains_any(*terms: str) -> Callable[[str], bool]:
    return lambda answer: any(term in _normalize(answer) for term in terms)


def _answers_with(found: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda answer: found(answer) and not is_non_answer(answer)


def _forty_five(answer: str) -> bool:
    text = _normalize(answer)
    return "45" in text or "forty-five" in text


def _five_to_seven(answer: str) -> bool:
    text = _normalize(answer)
    return ("five" in text and "seven" in text) or ("5" in text and "7" in text)


def _mentions_ten(answer: str) -> bool:
    return _TEN.search(_normalize(answer)) is not None


SUCCESS_CHECKS: tuple[Callable[[str], bool], ...] = (
    _forty_five,
    _forty_five,
    _five_to_seven,
    _contains_any("verification", "twelve", "12"),
    _contains_any("two-year", "two year", "2-year", "2 year"),
    _contains_any("out of stock"),
    _contains_any("20%", "20 percent", "twenty percent"),
    _answers_with(_contains_any("expedited", "two-day", "2-day")),
    _answers_with(_mentions_ten),
    _answers_with(lambda answer: "out of stock" in _normalize(answer) and _mentions_ten(answer)),
)
