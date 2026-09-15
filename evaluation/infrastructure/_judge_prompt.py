JUDGE_SYSTEM_PROMPT = """You are an impartial judge comparing two responses to the same query.

Score EACH response independently on these four dimensions, 1-5 each, per this rubric:
- coherence: 1 = disjointed or self-contradictory, 3 = holds together but has rough \
transitions, 5 = reads as a single well-organized answer
- relevance: 1 = misses the actual question asked, 3 = addresses the question but \
includes significant off-topic material, 5 = tightly focused on what was actually asked
- completeness: 1 = leaves out information the query clearly needed, 3 = covers the \
main point but skips secondary detail the query implied it wanted, 5 = covers \
everything the query needed at the depth it implied
- groundedness: 1 = asserts specific checkable facts with no support anywhere, 3 = \
mixes grounded and unverifiable claims, 5 = every specific checkable claim traces to \
its grounding source. If a REFERENCE PASSAGE is given below, score groundedness for \
BOTH responses against that reference passage, not against either response's own \
context section -- a response can retrieve the wrong material, generate a fluent \
answer that traces perfectly to that wrong material, and still be factually ungrounded \
against what's actually true, which grading against a response's own context can never \
catch. If no reference passage is given, fall back to judging each response's \
groundedness against its own context section instead. If a response's own context \
section reads "(none provided)", that \
response had nothing to ground itself in -- score it on whether it hedges appropriately \
for having no context (5) versus asserting specific checkable facts anyway with nothing \
behind them (1), not as an automatic floor and not by guessing whether its claims \
happen to be true.

Each response has its own context section above it, used for the hallucination check \
below -- judge whether a claim is *unverifiable* against that response's OWN context \
only, never the other response's; groundedness itself follows the reference-passage \
rule above when a reference passage is given. Score both responses on all four \
dimensions before comparing them to each other. Also flag, per response, any specific \
claim you cannot verify against the query or that response's own context -- a real \
flagged claim, never a bare "seems fine" or "seems off."

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"response_a": {"coherence": <int>, "relevance": <int>, "completeness": <int>, \
"groundedness": <int>, "unverifiable_claims": [<str>, ...]}, "response_b": \
{"coherence": <int>, "relevance": <int>, "completeness": <int>, "groundedness": \
<int>, "unverifiable_claims": [<str>, ...]}}
"""


def build_judge_user_message(
    query: str,
    response_a: str,
    response_b: str,
    context_a: str,
    context_b: str,
    reference_context: str = "",
) -> str:
    # Shared by OllamaJudge and ClaudeJudge so the two judges stay
    # comparable (docs/evaluation/COMPARISON_METHODOLOGY.md's self-vs-self
    # ablation design depends on that) -- previously hand-duplicated in both
    # files, which let a prompt edit to one silently drift from the other.
    if reference_context:
        reference_block = (
            f"Reference passage (the actual correct source material -- score BOTH "
            f"responses' groundedness against this, not against either response's own "
            f"context section below):\n{reference_context}\n\n"
        )
        context_label = (
            "use this for the unverifiable-claims check only -- for groundedness, "
            "see the reference passage above instead"
        )
    else:
        reference_block = ""
        context_label = (
            "use this to judge groundedness -- a claim that doesn't trace to this "
            "is unverifiable"
        )
    return (
        f"Query: {query}\n\n"
        f"{reference_block}"
        f"Context for Response A ({context_label}):\n"
        f"{context_a or '(none provided)'}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Context for Response B ({context_label}):\n"
        f"{context_b or '(none provided)'}\n\n"
        f"Response B:\n{response_b}"
    )
