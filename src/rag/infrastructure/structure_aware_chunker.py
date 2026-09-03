import re

import tiktoken

from src.rag.domain.ports import Chunker
from src.rag.infrastructure.sentence_based_chunker import SentenceBasedChunker

_HEADING = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)
_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)


def _split_into_sections(text: str) -> list[str]:
    """Split on Markdown heading lines, keeping each heading with the
    content that follows it up to the next heading. Code fences are matched
    and treated as opaque before heading-matching runs, so a heading-looking
    line inside a code block (e.g. a shell comment starting with #) is never
    mistaken for a real section break.
    """
    fence_spans = [m.span() for m in _CODE_FENCE.finditer(text)]

    def _inside_a_fence(pos: int) -> bool:
        return any(start <= pos < end for start, end in fence_spans)

    heading_starts = [m.start() for m in _HEADING.finditer(text) if not _inside_a_fence(m.start())]
    if not heading_starts:
        return [text]

    sections = []
    if heading_starts[0] > 0:
        sections.append(text[: heading_starts[0]])
    for i, start in enumerate(heading_starts):
        end = heading_starts[i + 1] if i + 1 < len(heading_starts) else len(text)
        sections.append(text[start:end])
    return [s for s in sections if s.strip()]


def _split_heading(section: str) -> tuple[str | None, str]:
    """Pull a leading Markdown heading line off a section, if it has one.
    Sections built by _split_into_sections always start at a heading's own
    position, so the heading is exactly the section's first line."""
    stripped = section.strip()
    first_line, _, rest = stripped.partition("\n")
    if _HEADING.match(first_line):
        return first_line, rest
    return None, stripped


class StructureAwareChunker(Chunker):
    def __init__(self, chunk_size_tokens: int = 512) -> None:
        self._chunk_size = chunk_size_tokens
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._fallback = SentenceBasedChunker(chunk_size_tokens)

    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []

        sections = _split_into_sections(text)
        chunks: list[str] = []
        for section in sections:
            fits = len(self._encoding.encode(section)) <= self._chunk_size
            if _CODE_FENCE.search(section) or fits:
                # A section containing a fenced code block is kept whole
                # even if it exceeds chunk_size_tokens -- a broken code
                # block is a worse failure than one oversized chunk. This
                # also covers the "no heading structure at all" case: when
                # _split_into_sections finds zero real headings, it returns
                # the whole document as one section, which lands here and
                # gets either kept whole (if it fits or contains a fence) or
                # sentence-chunked via the fallback below -- the same
                # fence-protection this loop already gives real sections,
                # rather than a separate, fence-unaware special case that
                # bypassed it.
                chunks.append(section.strip())
            else:
                # SentenceBasedChunker has no notion of a heading, so
                # handing it the whole section would leave the heading only
                # in the first sub-chunk -- every later sub-chunk of a long
                # section would be an orphaned continuation with no
                # structural context, defeating the point of a
                # structure-*aware* strategy. Split the heading off first
                # and prepend it back onto every sub-chunk explicitly.
                heading, body = _split_heading(section)
                sub_chunks = self._fallback.chunk(body)
                if heading is None:
                    chunks.extend(sub_chunks)
                else:
                    chunks.extend(f"{heading}\n{sub}" for sub in sub_chunks)
        return chunks
