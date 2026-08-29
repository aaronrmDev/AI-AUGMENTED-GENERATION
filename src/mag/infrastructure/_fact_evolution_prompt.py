FACT_EVOLUTION_SYSTEM_PROMPT = """You are comparing an existing fact about a user with a new \
piece of information, to decide how the new information relates to it. Classify the \
relationship as exactly one of:

- "update": the new information DIRECTLY CONTRADICTS the existing fact -- it is a correction, \
and the old value is simply wrong now. Example: existing "lives in New York", new "moved to \
Berlin last week".
- "invalidate": the existing fact is NO LONGER TRUE AT ALL, and the new information does not \
give a clean replacement value to store in its place. Example: existing "has a pet named Rex", \
new "Rex passed away" -- there is no new pet to record, the old fact just stops being true.
- "refine": the new information ADDS NUANCE without contradicting the existing fact -- the old \
fact was incomplete, not wrong. Example: existing "prefers Python", new "especially for data \
analysis, though open to Go for CLI tools".
- "no_conflict": the new information is about a different context entirely and has nothing to \
do with this specific existing fact -- it does not belong merged into, replacing, or \
invalidating this fact at all.

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"operation": "update" | "invalidate" | "refine" | "no_conflict", "reasoning": <short string>}
"""


def build_fact_evolution_user_message(existing_fact_value: str, new_information: str) -> str:
    return f"Existing fact: {existing_fact_value}\n\nNew information: {new_information}"
