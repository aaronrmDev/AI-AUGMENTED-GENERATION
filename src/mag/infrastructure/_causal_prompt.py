import json
from typing import Any

CAUSAL_SYSTEM_PROMPT = """You are scoring a list of episodic memories (past interactions) for \
how much each one describes a cause-effect chain relevant to a query -- an error trace, a root \
cause, or a fix, not just a topically-related episode. A query about "why did the deployment \
fail" is best answered by episodes that actually contain the failure and its cause, not every \
episode that merely mentions "deployment."

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"scores": [{"episode_index": <int, 1-based, matching the numbered list below>, \
"score": <float, 0.0-1.0>}, ...]}
Include exactly one entry per episode listed below, in any order.
"""


def build_causal_user_message(query: str, episodes: list[dict[str, Any]]) -> str:
    numbered = "\n\n".join(
        f"Episode {i} ({episode.get('timestamp', 'unknown time')}):\n"
        f"{json.dumps(episode['content'], sort_keys=True)}"
        for i, episode in enumerate(episodes, start=1)
    )
    return f"Query: {query}\n\nEpisodes:\n\n{numbered}"
