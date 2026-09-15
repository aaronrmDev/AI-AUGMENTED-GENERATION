from pathlib import Path

from evaluation.scenarios.loader import load_scenario

_FIXTURE_DIR = (
    Path(__file__).parent.parent.parent / "evaluation" / "scenarios" / "_fixtures" / "smoke-test"
)
_CHUNKING_STRATEGIES_DIR = (
    Path(__file__).parent.parent.parent
    / "evaluation"
    / "scenarios"
    / "rag-chunking-strategies"
)


def test_load_scenario_reads_name_and_questions():
    scenario = load_scenario(_FIXTURE_DIR)

    assert scenario.name == "smoke-test"
    assert len(scenario.questions) == 2
    assert scenario.questions[0].question == "What is this fixture for?"
    assert scenario.questions[0].success_criterion == "answer mentions 'fixture' or 'test'"


def test_load_scenario_resolves_the_corpus_directory():
    scenario = load_scenario(_FIXTURE_DIR)

    assert scenario.corpus_dir == _FIXTURE_DIR / "corpus"


def test_load_scenario_defaults_gold_passage_to_empty_string_when_absent():
    # smoke-test's queries.yaml predates gold_passage (#147) -- every
    # question must still load validly with a defaulted, not missing, value.
    scenario = load_scenario(_FIXTURE_DIR)

    assert all(q.gold_passage == "" for q in scenario.questions)


def test_load_scenario_reads_gold_passage_when_present():
    # rag-chunking-strategies/queries.yaml carries a real gold_passage per
    # question (#147) -- this is the actual production data, not a fixture,
    # so this test also guards against that YAML file's own shape drifting.
    scenario = load_scenario(_CHUNKING_STRATEGIES_DIR)

    assert len(scenario.questions) == 5
    assert all(q.gold_passage for q in scenario.questions)
    assert "512 tokens with 10% overlap" in scenario.questions[0].gold_passage
