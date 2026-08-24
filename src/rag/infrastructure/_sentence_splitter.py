import re

# Splits after ., !, or ? only when followed by whitespace and then an
# uppercase letter or end-of-string -- avoids splitting on a decimal point
# (3.14) or a mid-sentence abbreviation followed by a lowercase continuation.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z]|$)")


def split_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
