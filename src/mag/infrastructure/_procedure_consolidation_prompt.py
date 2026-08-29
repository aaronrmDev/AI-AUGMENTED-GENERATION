PROCEDURE_CONSOLIDATION_SYSTEM_PROMPT = """You are reflecting on a batch of episodic memories \
(past interactions) from a single conversation session, looking for a REPEATED task pattern \
-- the same kind of task attempted more than once -- and the workflow that made it succeed.

Extract a procedure only when the SAME task pattern appears at least twice in the batch with a \
successful outcome, using a recognizably similar sequence of steps each time. A single \
occurrence of a task, even a successful one, is not a repeated pattern -- don't invent a \
procedure from one example. If the episodes don't support a confident, repeated pattern, an \
empty list is a valid, honest answer.

Respond with ONLY this JSON shape, no other text, no markdown fencing:
{"procedures": [{"task_pattern": <str>, "workflow": {"steps": [<str>, ...]}, \
"success_rate": <float, 0.0-1.0>}, ...]}
"""
