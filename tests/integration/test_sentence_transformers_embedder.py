def test_embed_returns_a_384_dimensional_vector(embedding_model):
    result = embedding_model.embed("a sentence to embed")
    assert len(result) == 384
    assert all(isinstance(v, float) for v in result)


def test_similar_sentences_embed_closer_than_dissimilar_ones(embedding_model):
    import math

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b)

    base = embedding_model.embed("the cat sat on the mat")
    similar = embedding_model.embed("a cat was sitting on a mat")
    different = embedding_model.embed("quarterly financial earnings report")

    assert cosine_similarity(base, similar) > cosine_similarity(base, different)
