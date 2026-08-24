from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ScenarioQuestion:
    question: str
    success_criterion: str


@dataclass(frozen=True)
class Scenario:
    name: str
    corpus_dir: Path
    questions: list[ScenarioQuestion]


def load_scenario(scenario_dir: Path) -> Scenario:
    queries_path = scenario_dir / "queries.yaml"
    data = yaml.safe_load(queries_path.read_text(encoding="utf-8"))
    questions = [
        ScenarioQuestion(question=q["question"], success_criterion=q["success_criterion"])
        for q in data["questions"]
    ]
    return Scenario(name=data["name"], corpus_dir=scenario_dir / "corpus", questions=questions)
