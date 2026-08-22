# Quantitative Comparison Template

Copy this table into a GitHub issue comment (or export the same columns to CSV) once per Story that reaches its "real measured result documented" DoD item. One filled-in copy per model tested, since the whole point of the ablation design in `docs/evaluation/COMPARISON_METHODOLOGY.md` is that the model stays fixed within a single comparison and only the RAG/CAG/MAG usage varies — don't average across models into one row, since that's exactly the model-versus-model comparison this project deliberately isn't running.

## How to fill this in

Run the baseline row first — same model, same input set, RAG/CAG/MAG all switched off — before running any treatment row, so the treatment rows have something real to be a delta against. Every input set used across a Story's rows must be identical; changing the questions between the baseline and treatment runs invalidates the comparison before it starts. Task success is defined by whatever this specific Story's issue body states as its success criterion (a test suite passing, an exact-match, a retrieval hit) — write down which one you used in the Notes column, since "task success" means something different for a chunking-strategy Task than for a CRAG Story.

## Template

| Run | RAG | CAG | MAG | Model | Input tokens | Output tokens | Latency (p50 / p95) | Task success | Δ vs. baseline | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | ✗ | ✗ | ✗ | | | | | | — | |
| Treatment | ✓/✗ per technique | ✓/✗ | ✓/✗ | | | | | | | |

- **RAG / CAG / MAG columns** — mark which paradigms were active for this run. For an Individual-Techniques Story, only one of the three is ever ✓. For a Combinations Story, mark every constituent paradigm the combination actually spans (a same-paradigm archetype like CAG's "Long Context Champion" still only marks CAG; a cross-paradigm Story like State-Aware RAG marks both RAG and MAG).
- **Model** — the exact model and quantization/serving config (e.g. "Qwen3.8-27B, vLLM 0.X on ROCm, FP8"). Two runs of "the same model" at different quantization levels are not the same comparison — write down the config every time, not just the model name.
- **Input / Output tokens** — read directly from the serving engine's own usage reporting, not estimated.
- **Latency** — report both p50 and p95 across at minimum 5 repeated runs of the same input set; a single-run latency number is noise, not a measurement.
- **Task success** — a percentage or a pass/fail count against the Story's stated success criterion, not a subjective read.
- **Δ vs. baseline** — the actual delta (e.g. "−38% output tokens, −12% p50 latency, task success unchanged at 94%"), which is the number that actually answers "did this technique earn its complexity."
- **Notes** — corpus/input-set used, success-criterion definition, anything about the run that a reader would need to reproduce it or to doubt it appropriately.

## What this template does not replace

This sheet only carries numbers a token counter or a stopwatch produces without judgment. Whether the *quality* of the output held up under a technique that, say, compressed context by 50% is a separate question this sheet cannot answer on its own — see `docs/evaluation/qualitative-rubric.md` for that half, and read both together before checking a Story's DoD box, per the closing section of `docs/evaluation/COMPARISON_METHODOLOGY.md`.
