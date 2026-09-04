from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScenarioQuestion:
    question: str
    success_criterion: str
    # A run-independent ground-truth excerpt for this question, used as the
    # judge's groundedness reference instead of either arm's own retrieved
    # context (#147) -- optional because most existing scenarios' queries.yaml
    # files predate this field and have nothing to put here.
    gold_passage: str = ""


@dataclass(frozen=True)
class Scenario:
    name: str
    corpus_dir: Path
    questions: list[ScenarioQuestion]


def load_scenario(scenario_dir: Path) -> Scenario:
    queries_path = scenario_dir / "queries.yaml"
    data = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    questions = [
        ScenarioQuestion(
            question=q["question"],
            success_criterion=q["success_criterion"],
            gold_passage=q.get("gold_passage", ""),
        )
        for q in data["questions"]
    ]
    return Scenario(name=data["name"], corpus_dir=scenario_dir / "corpus", questions=questions)
