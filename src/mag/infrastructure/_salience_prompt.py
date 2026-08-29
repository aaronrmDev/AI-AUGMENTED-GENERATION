import json
from typing import Any

SALIENCE_SYSTEM_PROMPT = """You are rating a single episodic memory (one turn or event from a \
conversation) for how salient it is -- how much it looks like a critical decision, a failure, \
or an error, rather than a routine turn. A routine question-and-answer exchange is low \
salience; a decision that changes future behavior, or a failure/error/exception, is high \
salience.

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"salience_score": <float, 0.0-1.0>}
"""


def build_salience_user_message(content: dict[str, Any]) -> str:
    return f"Episode:\n\n{json.dumps(content, sort_keys=True)}"
