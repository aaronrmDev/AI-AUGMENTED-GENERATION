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
