import json
import math

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.errors import ClassificationFailed
from src.orchestration.domain.ports import QueryClassifier
from src.rag.domain.ports import ChatModel

_PROMPT_TEMPLATE = (
    "You route a user's question to the knowledge sources able to answer it. "
    "For each source, give your confidence from 0 to 1 that answering the "
    "question needs it. A question can need more than one source.\n"
    "- cag: stable reference knowledge that rarely changes and is asked about "
    "often, such as policies, manuals, guides, or a codebase.\n"
    "- mag: this user's own conversation history, preferences, or things they "
    "asked to have remembered.\n"
    "- rag: fresh or changing external information, such as today's data, "
    "recent changes, live figures, or news.\n"
    'Respond with ONLY a JSON object of the form {{"cag": 0.0, "mag": 0.0, '
    '"rag": 0.0}} and nothing else.\n\n'
    "Question: {query}"
)


def _parse_scores(raw: str) -> dict[Paradigm, float] | None:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scores: dict[Paradigm, float] = {}
    for paradigm in PARADIGM_ORDER:
        value = data.get(paradigm.value)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        scores[paradigm] = float(value)
    return scores


class LlmQueryClassifier(QueryClassifier):
    def __init__(self, chat_model: ChatModel) -> None:
        self._chat_model = chat_model
        # Counted, not hidden: the router report states how often the real
        # model's reply was unusable, the same failure #149 found in the judge.
        self.parse_failures = 0

    async def score(self, query: str, query_embedding: list[float]) -> dict[Paradigm, float]:
        raw = await self._chat_model.complete(_PROMPT_TEMPLATE.format(query=query))
        scores = _parse_scores(raw)
        if scores is None:
            self.parse_failures += 1
            # Raised rather than returned as neutral scores: 0.5 everywhere
            # only lands inside the router's uncertainty band at one threshold.
            raise ClassificationFailed(f"unparseable classifier reply: {raw[:120]!r}")
        return scores
