"""Cascade tier thresholds for the orchestration meta-layer, measured with the
real MiniLM embedder (sentence-transformers all-MiniLM-L6-v2) on 2026-09-13.

Shared by the integration tests (tests/integration/orchestration_env.py) and
the evaluation runner (evaluation/scenarios/orchestration-meta-layer/), so the
numbers the report cites are the numbers the tests pin. They are properties of
this embedder on this project's support-bot corpus, not general defaults --
which is why src/ gives the tiers no default thresholds at all.

Measured cosine similarities:
  CAG must hit:  "What changed in the return policy today?" vs. the thirty-day policy 0.5771,
                 "What is the return policy for unopened items?" vs. the same     0.7700
  CAG must miss: "How long does standard shipping take?" vs. the same             0.2833
  MAG must hit:  "What shipping speed did I say I prefer?" vs. the shipping fact  0.5202
  MAG must miss: "What is the return policy for unopened items?" vs. the same     0.3290

Hit = the midpoint of the lowest must-hit and highest must-miss score, floored to
two decimals. Partial = the midpoint of the must-miss score and the hit threshold,
so a clearly unrelated query can never become a PARTIAL match and carry the
superseded policy into its context.
"""

CAG_HIT = 0.43
CAG_PARTIAL = 0.35
MAG_HIT = 0.42
MAG_PARTIAL = 0.37
