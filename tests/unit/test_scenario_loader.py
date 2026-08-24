from pathlib import Path

from evaluation.scenarios.loader import load_scenario

_FIXTURE_DIR = Path(__file__).parent.parent.parent / "evaluation" / "scenarios" / "_fixtures" / "smoke-test"


def test_load_scenario_reads_name_and_questions():
    scenario = load_scenario(_FIXTURE_DIR)

    assert scenario.name == "smoke-test"
    assert len(scenario.questions) == 2
    assert scenario.questions[0].question == "What is this fixture for?"
    assert scenario.questions[0].success_criterion == "answer mentions 'fixture' or 'test'"


def test_load_scenario_resolves_the_corpus_directory():
    scenario = load_scenario(_FIXTURE_DIR)

    assert scenario.corpus_dir == _FIXTURE_DIR / "corpus"
