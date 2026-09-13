import uuid


class SessionNotFound(Exception):
    def __init__(self, session_id: uuid.UUID) -> None:
        super().__init__(f"no session {session_id} is visible under the current tenant")
        self.session_id = session_id


class ClassificationFailed(Exception):
    """A query classifier could not produce usable scores -- an unparseable
    model reply, for example. UnifiedAnswerQuestion routes such a query with
    paradigm_router.fallback_decision()."""


class QueryExceedsBudget(Exception):
    def __init__(self, query_tokens: int, query_slice: int) -> None:
        super().__init__(
            f"the question is {query_tokens} tokens; the Query slice allows {query_slice}"
        )
        self.query_tokens = query_tokens
        self.query_slice = query_slice
