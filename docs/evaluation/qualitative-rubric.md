# Qualitative Judge Rubric

This rubric exists to catch something a token count alone can't: a technique that cut token usage in half might have done it by being genuinely efficient, or by quietly giving worse answers, and only a quality check tells those two apart. Per `docs/evaluation/COMPARISON_METHODOLOGY.md`, it's meant to be run alongside the quantitative template, as that sheet's companion rather than its replacement. The design goal here is reproducibility, not sophistication: the same judge model, the same rubric wording, and the same input pair (baseline output vs. treatment output) should produce close to the same verdict whenever someone re-runs it, which is what makes a claimed quality improvement checkable rather than a one-time impression nobody else can verify.

## Judge model

Claude (Sonnet 5 or Opus 5, per `docs/evaluation/COMPARISON_METHODOLOGY.md`'s roster) is the default judge, specifically because it isn't one of the models being ablated — a model judging its own output, or a close sibling's, has no independent check on it. If Claude itself is ever the subject of a comparison (its own baseline-vs-treatment run), use Gemini as the judge for that specific comparison instead, for the same reason.

`OllamaJudge` (`evaluation/infrastructure/ollama_judge.py`) is the local fallback for exactly the situation this independence rule is meant to prevent: when the Anthropic account has no credit balance to run `ClaudeJudge` live, a comparison can still be judged for free against a local Ollama model instead. That fallback only holds to the independence rule if the judge model is a genuinely different family from whatever model generated the treatment being judged — the RAG chunking-strategies batch's original run violated this (qwen3.5 judging its own qwen3.5-generated treatment) and flagged the self-grading-bias risk explicitly in every report's notes field; its `#147` re-measurement fixed it by re-running with `OllamaJudge` configured against `llama3.1:8b` instead, a different model family (Meta, not Alibaba) from the qwen3.5 model under test. Any future comparison reaching for `OllamaJudge` needs to make the same check: the judge model and the model being ablated must not be the same family, or the judge carries the same bias `OllamaJudge`'s own header comment warns about.

## Judge prompt structure

Give the judge model exactly this, every time — the fixed structure is what makes scores comparable across runs:

1. The original query or task the baseline and treatment both responded to.
2. A reference passage, if the scenario supplies one (`reference_context` on `Judge.score()`) — a run-independent excerpt of the actual correct source material, distinct from anything either arm itself retrieved. Omitted entirely when a scenario doesn't supply one, which is still most scenarios today (see below).
3. The context the baseline actually retrieved (or an explicit "(none provided)" if it retrieved nothing), followed by the baseline output — labeled "Response A" (never labeled "baseline," so the judge can't anchor on which one is supposed to be better).
4. The context the treatment actually retrieved, followed by the treatment output — labeled "Response B."
5. The four dimensions below, each with its 1-5 scale definition, and an instruction to score both responses on each dimension independently before comparing them.
6. An instruction to flag, in a free-text field, anything either response asserts that the judge cannot verify against the query or *that response's own context* — never the other response's — this is the hallucination check, and it needs to be a specific flagged claim, not a bare "seems fine" or "seems off."

Each response gets its own context section (`context_a`/`context_b` on `Judge.score()`), not one shared context for both — scoring both arms against a single context structurally penalizes whichever arm didn't supply it, even when its own claims were genuinely grounded in what it actually retrieved (see `#148`). That per-arm context still drives the hallucination check in step 6 above, unchanged by the reference passage in step 2. Groundedness (step 5's fourth dimension) is different: when a reference passage is given, both arms' groundedness is scored against that shared reference instead of each arm's own context, precisely because an arm can retrieve the wrong material, answer fluently and consistently with that wrong material, and still be scored perfectly "grounded" against context that was itself never checked against anything true (`#147`) — a reference passage is what makes groundedness comparable across two runs that retrieved genuinely different material, the way the five RAG chunking-strategy reports now are. A scenario that never supplies a reference passage falls back to the original per-arm-context grounding for that dimension too, exactly as before `#147` existed.

## Scoring dimensions

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| **Coherence** | Response is disjointed or self-contradictory | Response holds together but has rough transitions or an unclear structure | Response reads as a single, well-organized answer |
| **Relevance** | Response misses the actual question asked | Response addresses the question but includes significant off-topic material | Response is tightly focused on what was actually asked |
| **Completeness** | Response leaves out information the query clearly needed | Response covers the main point but skips secondary detail the query implied it wanted | Response covers everything the query needed, at the depth the query implied |
| **Groundedness** | Response asserts specific, checkable facts with no support anywhere | Response mixes grounded and unverifiable claims | Every specific, checkable claim in the response traces to its grounding source |

Score both Response A and Response B on all four dimensions before writing anything about which is better — scoring them independently first is what stops the comparison from collapsing into "B feels more thorough, therefore B wins on everything."

## Recording a result

Paste the judge's four-dimension scores for both responses, plus any flagged unverifiable claims, into the same GitHub issue comment as the quantitative template's table. A qualitative result on its own, without the quantitative delta next to it, doesn't answer "was this technique worth its complexity" — it only answers "did quality hold up," which is half the question `docs/evaluation/COMPARISON_METHODOLOGY.md` sets out to close.

## Why this rubric, not a single overall score

A single 1-10 "how good is this" number collapses four different failure modes into one dial, and a technique that's perfectly coherent but ungrounded (confidently wrong) fails very differently from one that's grounded but incomplete (correctly incomplete) — collapsing those into the same low overall score would make two genuinely different problems look identical to whoever reads the result later. Keeping the four dimensions separate is what lets a reader tell which failure mode actually happened, rather than just that something scored low.
