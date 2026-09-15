import math

from src.orchestration.domain.entities import PARADIGM_ORDER, Paradigm
from src.orchestration.domain.paradigm_router import decide
from src.orchestration.infrastructure.prototype_query_classifier import (
    PrototypeQueryClassifier,
)
from src.orchestration.infrastructure.routing_exemplars import DEFAULT_ROUTING_EXEMPLARS


async def test_real_embeddings_produce_well_formed_scores_for_every_exemplar(embedding_model):
    classifier = PrototypeQueryClassifier(embedding_model)
    for exemplar in DEFAULT_ROUTING_EXEMPLARS:
        scores = await classifier.score(exemplar.query, embedding_model.embed(exemplar.query))
        assert set(scores) == set(PARADIGM_ORDER)
        assert all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores.values())


async def test_a_lightly_reworded_exemplar_routes_to_include_its_own_label(embedding_model):
    # Accuracy on held-out queries is measured in the router report, not
    # gated here -- gating on it would invite tuning exemplars to pass.
    classifier = PrototypeQueryClassifier(embedding_model, k=3)
    reworded = [
        ("Please tell me what the employee handbook says about vacation days.", Paradigm.CAG),
        ("Remind me, which dietary restrictions did I mention?", Paradigm.MAG),
        ("Right now, what is the inventory level for the blue running shoes?", Paradigm.RAG),
    ]
    for query, expected in reworded:
        decision = decide(await classifier.score(query, embedding_model.embed(query)))
        print(f"{query!r} -> {sorted(p.value for p in decision.paradigms)} {decision.scores}")
        assert expected in decision.paradigms
