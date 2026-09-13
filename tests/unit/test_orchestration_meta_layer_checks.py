"""The orchestration comparison's deterministic success checks, tested against
answers its live runs produced (evaluation/reports/orchestration-meta-layer-comparison*.md)
wherever such an answer exists."""
from collections.abc import Callable
from pathlib import Path

import pytest

from evaluation.scenarios.loader import load_scenario
from evaluation.scenarios.orchestration_meta_layer_checks import SUCCESS_CHECKS, is_non_answer

_SCENARIO_DIR = (
    Path(__file__).resolve().parents[2] / "evaluation/scenarios/orchestration-meta-layer"
)


def _check(question_number: int) -> Callable[[str], bool]:
    return SUCCESS_CHECKS[question_number - 1]


def test_there_is_one_check_per_question_in_the_scenario():
    assert len(SUCCESS_CHECKS) == len(load_scenario(_SCENARIO_DIR).questions)


@pytest.mark.parametrize(
    ("question_number", "answer"),
    [
        (8, "You prefer expedited two-day shipping on every order."),
        (9, "You wear size 10 running shoes."),
        (10, "No, as of this morning, the blue running shoes are out of stock in sizes 9 and 10."),
    ],
)
def test_a_personal_question_answered_from_the_users_facts_succeeds(question_number, answer):
    assert _check(question_number)(answer)


@pytest.mark.parametrize(
    ("question_number", "answer"),
    [
        # Verbatim from the prototype-routed comparison: both passed the earlier
        # check, because "sizes 9 and 10" contains a 10.
        (
            10,
            "The provided context states that blue running shoes are out of stock in sizes 9 "
            "and 10, but it does not specify what your shoe size is.",
        ),
        (
            10,
            "The provided context does not contain the answer because it does not specify what "
            "your shoe size is, only that blue running shoes in sizes 9 and 10 are out of stock.",
        ),
        # Constructed the same way for the other two personal questions.
        (
            9,
            "The provided context doesn't mention your shoe size; it only says sizes 9 and 10 "
            "are out of stock.",
        ),
        (
            8,
            "The provided context does not mention your preference, though expedited shipping "
            "arrives within two business days.",
        ),
    ],
)
def test_a_non_answer_that_quotes_the_retrieved_context_fails(question_number, answer):
    assert not _check(question_number)(answer)


def test_capitals_and_curly_apostrophes_are_still_recognized_as_a_non_answer():
    assert is_non_answer("The context DOESN’T specify your size.")


@pytest.mark.parametrize(
    "answer",
    [
        "You wear size 10 running shoes.",
        "No, as of this morning, the blue running shoes are out of stock in sizes 9 and 10.",
        "Yes, there is a flash sale today that takes 20% off every backpack until midnight.",
    ],
)
def test_a_direct_answer_is_not_a_non_answer(answer):
    assert not is_non_answer(answer)


@pytest.mark.parametrize(
    ("question_number", "answer", "expected"),
    [
        (1, "The return window for unopened items is forty-five days of purchase.", True),
        (1, "The return window for unopened items is thirty days of purchase.", False),
        (2, "The return window was extended to forty-five days today.", True),
        (3, "Standard shipping takes five to seven business days.", True),
        (
            4,
            "To set up an account, provide your email address and choose a password of at least "
            "twelve characters.",
            True,
        ),
        (5, "The product warranty is two years.", True),
        (6, "As of this morning, the blue running shoes are out of stock in sizes 9 and 10.", True),
        (
            7,
            "Yes, there is a flash sale today that takes 20% off every backpack until midnight.",
            True,
        ),
    ],
)
def test_the_other_checks_judge_real_answers_as_before(question_number, answer, expected):
    assert _check(question_number)(answer) is expected
