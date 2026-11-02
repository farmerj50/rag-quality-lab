
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def test_embedding_semantic_similarity():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "The team performs regular quality audits.",
        "The company conducts compliance reviews.",
        "The customer purchased a new laptop."
    ]

    embeddings = model.encode(sentences)

    assert embeddings.shape == (3, 384)

    scores = cosine_similarity(embeddings)

    assert scores[0][1] > scores[0][2]
