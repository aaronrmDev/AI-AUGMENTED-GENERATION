from pathlib import Path

from evaluation.application.run_comparison import RunComparison
from evaluation.domain.entities import Answer
from evaluation.infrastructure.markdown_report import render
from evaluation.scenarios.loader import load_scenario
from tests.unit.evaluation_fakes import FakeJudge

_FIXTURE_DIR = (
    Path(__file__).parent.parent.parent / "evaluation" / "scenarios" / "_fixtures" / "smoke-test"
)


async def test_full_harness_pipeline_produces_a_readable_report():
    scenario = load_scenario(_FIXTURE_DIR)

    async def baseline(question: str) -> Answer:
        return Answer(
            text="a baseline answer with no fixture awareness", input_tokens=20, output_tokens=10
        )

    async def treatment(question: str) -> Answer:
        return Answer(
            text="a treatment answer mentioning this is a test fixture",
            input_tokens=20,
            output_tokens=8,
        )

    use_case = RunComparison(judge=FakeJudge(), repeat_count=2)
    result = await use_case.execute(
        scenario_name=scenario.name,
        model_config="qwen3.5, Ollama",
        success_criterion="see per-question criteria",
        questions=[q.question for q in scenario.questions],
        baseline=baseline,
        treatment=treatment,
        success_check=lambda q, a: True,
    )

    report = render(result)

    assert scenario.name in report
    assert "Quantitative" in report
    assert "Qualitative" in report
    assert len(result.judge_scores) == len(scenario.questions)
