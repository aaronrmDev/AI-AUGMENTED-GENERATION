import tiktoken

# cl100k_base, not a per-call parameter -- matches the encoding
# src/rag/infrastructure/compressing_retriever.py already committed to for
# the same kind of token-budget accounting, so a token count computed here
# and one computed there mean the same thing. get_encoding() is cached by
# tiktoken itself, so module-level reuse across calls costs nothing extra.
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))
