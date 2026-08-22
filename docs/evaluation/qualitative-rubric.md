# Qualitative Judge Rubric

Quantitative numbers can't tell you whether a technique that cut token usage in half did it by being efficient or by quietly giving worse answers — that's what this rubric exists to catch, and per `docs/evaluation/COMPARISON_METHODOLOGY.md` it's meant to be run alongside the quantitative template, never as a substitute for it. The design goal here is reproducibility, not sophistication: the same judge model, the same rubric wording, and the same input pair (baseline output vs. treatment output) should produce close to the same verdict whenever someone re-runs it, which is what makes a claimed quality improvement checkable rather than a one-time impression nobody else can verify.

## Judge model

Claude (Sonnet 5 or Opus 5, per `docs/evaluation/COMPARISON_METHODOLOGY.md`'s roster) is the default judge, specifically because it isn't one of the models being ablated — a model judging its own output, or a close sibling's, has no independent check on it. If Claude itself is ever the subject of a comparison (its own baseline-vs-treatment run), use Gemini as the judge for that specific comparison instead, for the same reason.

## Judge prompt structure

Give the judge model exactly this, every time — the fixed structure is what makes scores comparable across runs:

1. The original query or task the baseline and treatment both responded to.
2. The baseline output, labeled "Response A" (never labeled "baseline," so the judge can't anchor on which one is supposed to be better).
3. The treatment output, labeled "Response B."
4. The four dimensions below, each with its 1-5 scale definition, and an instruction to score both responses on each dimension independently before comparing them.
5. An instruction to flag, in a free-text field, anything either response asserts that the judge cannot verify against the query or any context provided — this is the hallucination check, and it needs to be a specific flagged claim, not a bare "seems fine" or "seems off."

## Scoring dimensions

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| **Coherence** | Response is disjointed or self-contradictory | Response holds together but has rough transitions or an unclear structure | Response reads as a single, well-organized answer |
| **Relevance** | Response misses the actual question asked | Response addresses the question but includes significant off-topic material | Response is tightly focused on what was actually asked |
| **Completeness** | Response leaves out information the query clearly needed | Response covers the main point but skips secondary detail the query implied it wanted | Response covers everything the query needed, at the depth the query implied |
| **Groundedness** | Response asserts specific facts with no support in the provided context, and those facts are checkable | Response mixes grounded and unverifiable claims | Every specific, checkable claim in the response traces to the provided context |

Score both Response A and Response B on all four dimensions before writing anything about which is better — scoring them independently first is what stops the comparison from collapsing into "B feels more thorough, therefore B wins on everything."

## Recording a result

Paste the judge's four-dimension scores for both responses, plus any flagged unverifiable claims, into the same GitHub issue comment as the quantitative template's table. A qualitative result on its own, without the quantitative delta next to it, doesn't answer "was this technique worth its complexity" — it only answers "did quality hold up," which is half the question `docs/evaluation/COMPARISON_METHODOLOGY.md` sets out to close.

## Why this rubric, not a single overall score

A single 1-10 "how good is this" number collapses four different failure modes into one dial, and a technique that's perfectly coherent but ungrounded (confidently wrong) fails very differently from one that's grounded but incomplete (correctly incomplete) — collapsing those into the same low overall score would make two genuinely different problems look identical to whoever reads the result later. Keeping the four dimensions separate is what lets a reader tell which failure mode actually happened, rather than just that something scored low.
