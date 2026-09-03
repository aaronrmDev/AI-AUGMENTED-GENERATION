REFINE_SYSTEM_PROMPT = """You are merging an existing fact about a user with new information that \
adds nuance to it, WITHOUT contradicting it -- the existing fact is not wrong, it is incomplete. \
Produce one richer fact_value that preserves everything true in the existing fact and folds in \
the new nuance, in a single natural sentence or short phrase (not a list, not two separate \
sentences glued together).

Example: existing fact "prefers Python", new information "especially for data analysis, though \
open to Go for CLI tools" -> merged fact_value "prefers Python, especially for data analysis, \
though open to Go for CLI tools".

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"merged_fact_value": <string>}
"""


def build_refine_user_message(existing_fact_value: str, new_information: str) -> str:
    return f"Existing fact: {existing_fact_value}\n\nNew information: {new_information}"
