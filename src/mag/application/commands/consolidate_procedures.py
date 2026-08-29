import json
import uuid
from typing import Any

from src.mag.application.commands.record_procedure import RecordProcedure
from src.mag.domain.entities import EpisodicMemory, ProceduralMemory
from src.mag.domain.ports import ProceduralMemoryRepository
from src.mag.infrastructure._llm_json import format_episodes_for_reflection, strip_markdown_fence
from src.mag.infrastructure._procedure_consolidation_prompt import (
    PROCEDURE_CONSOLIDATION_SYSTEM_PROMPT,
)
from src.rag.domain.ports import ChatModel

# Same reasoning as ConsolidateEpisodes's identical constant -- complete() has
# no forced JSON mode, so a malformed or fenced response is a real, observed
# possibility.
_MAX_REFLECTION_ATTEMPTS = 3


class ConsolidateProcedures:
    # MAG.md's "Self-Improving Agent" archetype (issue #24/#80) claims
    # Consolidation "analyzes which episodes succeeded and which failed"
    # and "extracts reusable Procedural memories" -- a mechanism
    # ConsolidateEpisodes (which only ever extracts semantic facts) never
    # implemented. This is that path, structurally mirroring
    # ConsolidateEpisodes closely enough to justify not merging the two,
    # but genuinely different in one respect ConsolidateEpisodes isn't:
    # ConsolidateEpisodes owns episodic_memory's consolidated_at lifecycle
    # (read-unconsolidated, reflect once, mark done -- a one-shot
    # operation per episode). Procedural extraction has no equivalent
    # one-shot semantics -- the SAME episode can legitimately contribute
    # to more than one future pattern detection as more repeats
    # accumulate over time -- so this command takes its episode batch as
    # an explicit parameter rather than fetching and marking its own
    # "unconsolidated" set. A caller composes this with
    # ConsolidateEpisodes over the same window explicitly (this project's
    # established explicit-parameters, no-hidden-orchestration
    # convention -- Batch C's retrieval strategies, Batch E's gating
    # strategies, Batch F's evolution operations all take the same
    # shape), rather than this command silently depending on
    # ConsolidateEpisodes not having already flipped consolidated_at on
    # the same batch.
    #
    # THE COMPOSITION HAZARD (Batch G adversarial review, HIGH severity):
    # ConsolidateEpisodes.execute() unconditionally calls mark_consolidated
    # on every episode it fetches, regardless of whether a fact was
    # extracted. If a caller composing the two commands builds this
    # method's episode batch by calling
    # EpisodicMemoryRepository.get_unconsolidated_by_session AFTER
    # ConsolidateEpisodes has already run over that window, the query
    # returns [] and this command silently returns [] right back (no
    # exception, no log) -- permanently skipping procedure extraction for
    # that batch. THE FIX: build the batch via the UNFILTERED
    # get_by_session (no consolidated_at predicate) instead of
    # get_unconsolidated_by_session whenever the same window is also
    # going through ConsolidateEpisodes -- get_by_session's result is
    # unaffected by mark_consolidated, so the fetch order relative to
    # ConsolidateEpisodes stops mattering. See
    # tests/integration/test_mag_combinations.py's
    # test_consolidate_procedures_composes_safely_with_consolidate_episodes_over_the_same_batch
    # for this pattern proven end-to-end, including a direct demonstration
    # that get_unconsolidated_by_session really does starve after
    # ConsolidateEpisodes runs.
    def __init__(
        self,
        procedural_memory_repository: ProceduralMemoryRepository,
        chat_model: ChatModel,
    ) -> None:
        self._record_procedure = RecordProcedure(procedural_memory_repository)
        self._chat_model = chat_model

    async def execute(
        self, tenant_id: uuid.UUID, user_id: uuid.UUID, episodes: list[EpisodicMemory]
    ) -> list[ProceduralMemory]:
        if not episodes:
            return []

        procedures_raw = await self._reflect(episodes)

        written: list[ProceduralMemory] = []
        for procedure in procedures_raw:
            recorded = await self._record_procedure.execute(
                tenant_id=tenant_id,
                user_id=user_id,
                task_pattern=procedure["task_pattern"],
                workflow=procedure["workflow"],
                success_rate=procedure["success_rate"],
            )
            written.append(recorded)
        return written

    async def _reflect(self, episodes: list[EpisodicMemory]) -> list[dict[str, Any]]:
        prompt = (
            f"{PROCEDURE_CONSOLIDATION_SYSTEM_PROMPT}\n\n"
            f"{format_episodes_for_reflection(episodes)}"
        )
        for _ in range(_MAX_REFLECTION_ATTEMPTS):
            response = await self._chat_model.complete(prompt)
            try:
                parsed = json.loads(strip_markdown_fence(response))
                procedures = parsed["procedures"]
                if not isinstance(procedures, list):
                    raise TypeError("'procedures' must be a list")
                return _validate_and_dedupe_procedures(procedures)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        # Exhausted retries: no procedures found, same "reflect once,
        # don't crash the caller" reasoning as ConsolidateEpisodes's
        # identical fallback.
        return []


def _validate_and_dedupe_procedures(procedures: list[Any]) -> list[dict[str, Any]]:
    # Same reasoning as ConsolidateEpisodes's _validate_and_dedupe_facts --
    # the outer {"procedures": [...]} envelope being valid JSON says
    # nothing about what's inside each element.
    validated: list[dict[str, Any]] = []
    seen_patterns: dict[str, int] = {}
    for item in procedures:
        if not isinstance(item, dict):
            raise TypeError("each procedure must be a JSON object")
        task_pattern = item["task_pattern"]
        workflow = item["workflow"]
        if not isinstance(task_pattern, str) or not task_pattern.strip():
            raise TypeError("task_pattern must be a non-empty string")
        # Non-empty, not just the right type -- an LLM confidently naming
        # a repeated task_pattern while returning workflow={} would
        # otherwise persist a stepless "procedure" with no error anywhere
        # downstream (RecordProcedure, the frozen ProceduralMemory
        # dataclass, and the Alembic schema all accept {} silently),
        # defeating the whole point of extracting a reusable workflow.
        if not isinstance(workflow, dict) or not workflow:
            raise TypeError("workflow must be a non-empty JSON object")
        # A dict containing a "steps" key is truthy regardless of what's
        # IN that list -- {"steps": []} (the system prompt's own
        # documented shape, PROCEDURE_CONSOLIDATION_SYSTEM_PROMPT's
        # {"steps": [<str>, ...]}) passes the check above untouched while
        # still being exactly as stepless as {} (review-caught: a
        # fix-wave re-review found the first version of this guard only
        # closed the literal {} case, not this one level deeper).
        steps = workflow.get("steps")
        if isinstance(steps, list) and not steps:
            raise TypeError("workflow.steps must not be an empty list")
        task_pattern = task_pattern.strip()
        # bool is a subclass of int in Python -- same guard
        # CaptureEpisode's salience validator and ConsolidateEpisodes's
        # confidence validator already use, applied here to success_rate.
        # Defaults to 1.0 (not RecordProcedure's own bare-call default of
        # 0.0) because this default only fires when the LLM omits the
        # field entirely, and the system prompt's own extraction
        # criterion requires a successful outcome each occurrence for a
        # pattern to be extracted at all -- mirrors ConsolidateEpisodes's
        # identically-reasoned confidence-defaults-to-1.0 idiom.
        raw_success_rate = item.get("success_rate", 1.0)
        if isinstance(raw_success_rate, bool) or not isinstance(raw_success_rate, (int, float)):
            raise TypeError("success_rate must be a number")
        success_rate = float(raw_success_rate)
        if not (0.0 <= success_rate <= 1.0):
            raise ValueError("success_rate out of [0.0, 1.0] range")
        record = {"task_pattern": task_pattern, "workflow": workflow, "success_rate": success_rate}
        # Last-wins on a duplicate task_pattern, same reasoning as
        # ConsolidateEpisodes's fact_key dedup -- RecordProcedure's
        # deterministic id means both would resolve to the same row
        # anyway.
        if task_pattern in seen_patterns:
            validated[seen_patterns[task_pattern]] = record
        else:
            seen_patterns[task_pattern] = len(validated)
            validated.append(record)
    return validated
