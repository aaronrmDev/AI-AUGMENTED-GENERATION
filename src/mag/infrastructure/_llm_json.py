import json
from typing import Any


def format_episodes_for_reflection(episodes: list[Any]) -> str:
    """Numbers each episode's content as JSON, oldest first. Shared by
    every reflection prompt that reasons over a raw episode batch
    (ConsolidateEpisodes, ConsolidateProcedures) -- a review finding
    (MAG Batch G) caught these as byte-for-byte duplicate functions in
    each command's own prompt module. A future formatting change
    (truncating long content, escaping a field) now has one place to
    land instead of two that can silently diverge."""
    numbered = "\n\n".join(
        f"Episode {i} ({episode.timestamp.isoformat()}):\n"
        f"{json.dumps(episode.content, sort_keys=True)}"
        for i, episode in enumerate(episodes, start=1)
    )
    return f"Episodes (oldest first):\n\n{numbered}"


def strip_markdown_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop the opening fence (optionally "```json") and a trailing
        # closing fence if present -- a model ignoring "no markdown
        # fencing" is exactly the kind of non-compliance #149 established
        # this project can't assume away.
        if lines and lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        stripped = "\n".join(lines)
    return stripped
