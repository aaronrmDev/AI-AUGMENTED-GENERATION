# Comparison Methodology

This document explains how this project measures whether RAG, CAG, and MAG actually deliver what `docs/architecture/RAG.md`, `CAG.md`, and `MAG.md` claim they deliver, and it exists because "we implemented it" and "we proved it works" are different claims, and this repository's own writing standard (`docs/governance/WRITING_STANDARDS.md`) treats the gap between them as a citation-discipline problem, not a formality. Every one of the 143 issues tracked across this project's six GitHub Projects boards (`rag-management`, `cag-management`, `mag-management`, and the three cross-paradigm boards) ends its Definition of Done with some form of "real measured result documented" rather than "implemented" alone, and this document is what makes that checkbox mean something specific and repeatable instead of a judgment call made once and never checked again.

## The comparison is self-versus-self, not model-versus-model

The obvious way to run this kind of evaluation is to line up several language models and ask which one is "best." That's not what this project needs, and running it that way would actively hide the thing worth measuring. The question this project exists to answer is narrower and more useful: for a given model, how much does adding RAG, CAG, or MAG — individually, in a pair, or in one of the larger named combinations already catalogued in the concept documents — actually change token burn, latency, and answer quality, relative to that same model with none of them switched on. The model is the control. RAG, CAG, and MAG are the only things that vary. A result that says "Gemma 4 with Cache-Warmed RAG uses 40% fewer output tokens than Gemma 4 alone, on this corpus, measured on this date" is a claim this project can stand behind. A result that says "Gemma 4 beats Qwen3.8-27B" answers a question nobody asked here and says nothing about whether this project's own architecture is earning its complexity.

Concretely, this means every technique or combination Story in the 143-issue tracker gets run twice against the same model on the same input set: once with that capability switched off (the baseline) and once with it switched on. The delta between those two runs — not either run in isolation — is the number that goes in the issue.

## The model roster, and what each one is for

Two roles, not one undifferentiated list of models: models this project actually self-hosts and optimizes against, and models this project only calls through an API as an external reference point. Conflating the two would blur exactly the distinction the ablation design depends on — a self-hosted model's token count is something this project's own CAG work can change; an API model's token count is a bill, useful for comparison, but not something PagedAttention or KV-cache eviction run against.

### Self-hosted (the actual ablation subjects)

Both served locally via **vLLM on ROCm**, targeting the project owner's AMD Radeon 7900 XTX (24GB VRAM) — see the reconciliation note in `docs/architecture/OVERVIEW.md`'s technology stack section for why this is vLLM-on-ROCm rather than the Ollama/`llama.cpp` path that was considered and ruled out.

- **Gemma 4** (12B / 26B-A4B / 31B sizes) — Google, released April 2, 2026, Apache 2.0, 256K native context. Chosen first, ahead of the context-window comparison below; kept in the roster because dropping an explicitly chosen model on a secondary criterion isn't this document's call to make unilaterally.
- **Qwen3.8-27B** — Alibaba, open-weight, Apache 2.0, released August 8, 2026, explicitly built to run on a single consumer GPU. 262,144-token native context, documented extension to 1,000,000 via YaRN — the better of the two self-hosted models on raw context, and also the stronger of the two on coding benchmarks (HumanEval, SWE-bench, LiveCodeBench, Aider) across every independent source checked, though those specific percentage figures came from secondary aggregator sites rather than a primary vendor benchmark page and should be treated as directionally right, not exact.

### API-based (reference points and judges)

- **DeepSeek V4** (V4-Pro, 1.6T MoE; V4-Flash, 284B MoE) — DeepSeek, MIT-licensed open weights, but at this scale not realistically self-hostable on a single 24GB card without heavy MoE offloading, so it's used via API here. 1M-token context, 384K max output — the largest output ceiling of any model in this roster, which matters directly for CAG: a model that can hold a large frozen context *and* generate a long answer from it gives a more honest signal about what long-context caching is actually worth. Leads raw coding benchmarks (SWE-bench Verified, LiveCodeBench) among the models checked.
- **Claude** (Sonnet 5 / Opus 5) — Anthropic, 1M-token context. Used as the primary qualitative judge (see below), not as an ablation subject.
- **Gemini** (3.1 Pro / 3.6 Flash) — Google, ~1.05M-token context, smaller max output (64–65.5K) than the other API models here. Used as a secondary qualitative reference, useful specifically because it's an independent model family from both the self-hosted models (Google, but a different generation and training run) and the primary judge.

All of the figures above were checked against current sources at the time this document was written (August 2026) rather than pulled from training-data recall, precisely because "Gemma 4," "Qwen3.8," and "DeepSeek V4" all postdate any language model's training cutoff by definition — see the citation list at the end of this document.

## What gets measured, and how each number earns trust

Two separate sheets, because they fail differently and mixing them hides that. See `docs/evaluation/quantitative-template.md` and `docs/evaluation/qualitative-rubric.md` for the actual reusable templates; this section is what each column *means* and why it's there.

### Quantitative: numbers that don't require judgment

Input tokens, output tokens, wall-clock latency, and an objective task-success signal (a test suite passing, an exact-match against a known answer, a retrieval hit against a known-relevant document — whatever the specific technique's Story defines as success, stated in that issue before the run happens, not chosen after seeing results that would flatter a number). None of these require a human or a model to form an opinion; they're read off the run directly. This is also where the "how much does it cost" and "how much energy does it burn" questions from earlier in this project's scoping actually get answered, since token count is the direct proxy for both — an approach that burns fewer tokens for the same task success is cheaper and lighter, full stop, without needing a separate carbon-accounting exercise this project has no way to run rigorously anyway.

### Qualitative: numbers that require judgment, made reproducible

Coherence, relevance, and hallucination rate aren't things a token counter can measure, but "we eyeballed it and it seemed fine" is exactly the kind of speculation this project's own writing standard rules out. The fix isn't to avoid qualitative measurement — it's to make the judgment reproducible: a fixed external judge model (Claude, per the roster above, specifically because it's the strongest available judge and is not one of the models being ablated, so it has no stake in either run), a written rubric with the same wording every time (`docs/evaluation/qualitative-rubric.md`), and the same prompt structure for every comparison. A judge score is still a judgment call, but it's the same judgment call applied consistently, which is what makes a claimed improvement checkable by someone who wasn't in the room when it was scored — they can re-run the same rubric against the same two outputs and expect to land near the same verdict.

## How a result gets from a run to an issue

Once a Story's baseline and treatment runs are both complete, the quantitative delta and the qualitative judge score both get pasted into that issue's GitHub thread, and the Story's "real measured result documented" DoD item gets checked only once both are present — a quantitative-only result without a quality check can't tell you whether a token-burn improvement came from the technique working or from the model quietly giving worse, shorter answers, and a qualitative-only result can't tell you whether an improvement is worth its cost. Neither sheet alone closes the loop this project's whole "0 speculation" framing exists to close; both together do.

## Sources

- [Gemma 4 by Google — Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/gemma-4-available-on-google-cloud)
- [Gemma 4: Byte for byte, the most capable open models — Google Blog](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Gemma 4 model card — Google AI for Developers](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Qwen/Qwen3.8-27B — Hugging Face](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Qwen 3.8-Max: Specs, Pricing, Benchmark Status — Yotta Labs](https://www.yottalabs.ai/post/qwen-3-8-max-release-date-specs-how-to-access-2026)
- [DeepSeek V4 Explained: V4-Pro 1.6T vs V4-Flash 284B](https://deepseek.ai/deepseek-v4)
- [DeepSeek officially launches V4-Pro — Quartz](https://qz.com/deepseek-v4-pro-official-launch-081326)
- [Gemini 3.1 Pro — Model Card, Google DeepMind](https://deepmind.google/models/model-cards/gemini-3-1-pro/)
- [Gemini 3.6 Flash — API Pricing & Benchmarks, OpenRouter](https://openrouter.ai/google/gemini-3.6-flash)
