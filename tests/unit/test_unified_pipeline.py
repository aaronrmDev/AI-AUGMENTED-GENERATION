import uuid

import pytest

from evaluation.scenarios.orchestration_meta_layer_thresholds import MAG_HIT, MAG_PARTIAL
from src.api import unified_pipeline
from src.orchestration.domain.entities import Paradigm
from src.orchestration.domain.errors import SessionNotFound
from tests.unit.rag_fakes import FakeChatModel, FakeEmbeddingModel, FakeVectorStore
from tests.unit.session_fakes import FakeChatSessionRepository


def _pipeline(sessions=None):
    return unified_pipeline.build_unified_pipeline(
        sessionmaker=object(),  # only a real query uses it; these tests never run one
        sessions=sessions or FakeChatSessionRepository(),
        embedding_model=FakeEmbeddingModel(),
        vector_store=FakeVectorStore(),
        chat_model=FakeChatModel(),
    )


def test_the_serving_thresholds_match_the_measured_ones():
    assert (unified_pipeline.MAG_HIT, unified_pipeline.MAG_PARTIAL) == (MAG_HIT, MAG_PARTIAL)


def test_the_cascade_serves_mag_and_rag_and_no_cag_tier():
    # LatencyCascade has no public accessor for its tiers; this reads the private map
    # rather than add one just for a test.
    assert set(_pipeline().cascade._tiers) == {Paradigm.MAG, Paradigm.RAG}


async def test_the_composed_path_refuses_a_session_before_answering():
    with pytest.raises(SessionNotFound):
        await _pipeline().answer_in_session.execute(
            uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "What is the return policy?"
        )
