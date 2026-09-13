import pytest

from src.orchestration.domain.similarity import cosine_similarity


def test_identical_vectors_score_one():
    assert cosine_similarity([0.6, 0.8], [0.6, 0.8]) == pytest.approx(1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_a_zero_vector_scores_zero_instead_of_dividing_by_zero():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_vectors_of_different_lengths_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0], [1.0])
