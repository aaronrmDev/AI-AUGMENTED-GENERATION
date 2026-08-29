from src.mag.domain.entities import GatingCandidate

# fact first, episode after, graph_node last -- MAG.md's own worked example:
# "places user preferences and task-critical facts first, adds supporting
# episodic context after." An unrecognized source_type falls back to
# len(_SOURCE_TYPE_PRIORITY) below, i.e. after graph_node, rather than
# raising a KeyError.
_SOURCE_TYPE_PRIORITY = {"fact": 0, "episode": 1, "graph_node": 2}


class HierarchicalAssembly:
    # Unlike its sibling gating strategies, this one never filters -- it's a
    # pure reordering step meant to run on an already-selected candidate
    # list (issue #55), putting the highest-priority information first so
    # the "lost in the middle" effect doesn't bury it.
    async def execute(
        self, candidates: list[GatingCandidate]
    ) -> list[GatingCandidate]:
        return sorted(
            candidates,
            key=lambda c: (
                _SOURCE_TYPE_PRIORITY.get(c.source_type, len(_SOURCE_TYPE_PRIORITY)),
                -c.score,
            ),
        )
