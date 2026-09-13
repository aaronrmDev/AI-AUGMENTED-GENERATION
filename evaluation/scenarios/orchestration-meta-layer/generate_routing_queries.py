"""Generates the router comparison's held-out query set with the local qwen3.5
model. Run once, only after the three classifiers, their cue lists, and their
exemplars are committed, so the evaluation phrasings are not written by the
author of the cues. Review every label by hand afterward; record any
correction in that entry's review_note instead of editing silently.

Usage (from the repository root):
    PYTHONPATH=. python evaluation/scenarios/orchestration-meta-layer/generate_routing_queries.py
"""
import asyncio
import re
from pathlib import Path

import ollama
import yaml

_OUT = Path(__file__).parent / "routing_queries.yaml"
_MODEL_ID = "qwen3.5"
_PER_ROUTE = 8
_ROUTES = {
    "cag": (
        "stable reference knowledge that rarely changes, such as a company policy, a "
        "product manual, a how-to guide, or an explanation of existing code"
    ),
    "rag": (
        "fresh or changing external information, such as today's figures, live status, "
        "recent news, or something that changed recently"
    ),
    "mag": (
        "only the user's own earlier conversation with the assistant, their stated "
        "preferences, or something they asked the assistant to remember"
    ),
    "mag,rag": (
        "BOTH the user's own earlier conversation or preferences AND fresh or changing "
        "external information, in the same question"
    ),
    "cag,rag": (
        "BOTH stable reference documentation AND whether that documented information has "
        "changed or is available right now, in the same question"
    ),
}
_CONCEPT_ONE = [
    ("What's our refund policy?", ["cag"]),
    ("What changed in the policy today?", ["rag"]),
    ("Continue where we left off yesterday", ["mag"]),
    ("Compare today's sales with last month", ["mag", "rag"]),
    ("Explain this code file", ["cag"]),
    ("What did I ask you to remember?", ["mag"]),
]
_PROMPT = (
    "Write {n} different requests that a user might send to an AI assistant used by an "
    "online store's customers and staff. Every request must need {description}. Vary the "
    "topic and the wording. Output one request per line, with no numbering and nothing else."
)
_LEADING_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


async def _generate(client: ollama.AsyncClient, description: str) -> list[str]:
    prompt = _PROMPT.format(n=_PER_ROUTE, description=description)
    response = await client.chat(model=_MODEL_ID, messages=[{"role": "user", "content": prompt}])
    content = response.message.content or ""
    lines = [_LEADING_MARKER.sub("", line).strip() for line in content.splitlines()]
    return [line for line in lines if len(line) > 10][:_PER_ROUTE]


async def _run() -> None:
    client = ollama.AsyncClient()
    entries: list[dict[str, object]] = []
    for labels, description in _ROUTES.items():
        generated = await _generate(client, description)
        print(f"{labels}: {len(generated)} generated")
        entries.extend(
            {"query": query, "expected": labels.split(","), "source": _MODEL_ID}
            for query in generated
        )
    entries.extend(
        {"query": query, "expected": expected, "source": "concept-1-table"}
        for query, expected in _CONCEPT_ONE
    )
    document = {"name": "orchestration-meta-layer-routing", "queries": entries}
    _OUT.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {len(entries)} queries to {_OUT}")


if __name__ == "__main__":
    asyncio.run(_run())
