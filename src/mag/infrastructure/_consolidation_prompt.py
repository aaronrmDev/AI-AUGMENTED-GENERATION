import json
from typing import Any

CONSOLIDATION_SYSTEM_PROMPT = """You are reflecting on a batch of episodic memories \
(past interactions) from a single conversation session, looking for durable facts \
worth remembering long-term.

Extract only generalized, timeless facts about the user's preferences, interests, \
or characteristics -- not a record of what happened in this specific conversation. \
"User asked about Python three times" is not itself a durable fact; "User's primary \
programming language is Python" is. If the episodes don't support a fact that \
confident, don't invent one -- an empty list is a valid, honest answer.

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"facts": [{"fact_key": <str>, "fact_value": <str>, "confidence": <float, 0.0-1.0>}, ...]}
"""


def build_consolidation_user_message(episodes: list[dict[str, Any]]) -> str:
    numbered = "\n\n".join(
        f"Episode {i} ({episode.get('timestamp', 'unknown time')}):\n"
        f"{json.dumps(episode['content'], sort_keys=True)}"
        for i, episode in enumerate(episodes, start=1)
    )
    return f"Episodes (oldest first):\n\n{numbered}"
