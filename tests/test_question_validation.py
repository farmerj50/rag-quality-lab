import pytest
from pydantic import ValidationError
from app.main import Question
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def test_rejects_invalid_top_k():
    with pytest.raises(ValidationError):
        Question(question='How are audits managed', top_k=11
                 )
        
def test_accepts_valid_top_k():
    payload = Question(question='How are audits managed', top_k=3
                       )
    assert payload.top_k == 3         
    
@pytest.mark.parametrize(
    "top_k",
    [1,2,3,4,9,10]
)
def test_valid_top_k_boundaries(top_k):
    payload = Question(question='How are audits managed', top_k=top_k
                       )
    assert payload.top_k == top_k
    
@pytest.mark.parametrize(
    "top_k",
    [11, 0, -1, 100]
)
def test_invalid_top_k_boundaries(top_k):
    with pytest.raises(ValidationError):
        Question(question='How are audits managed', top_k=top_k
                 )
def test_default_top_k():
    payload = Question(question='How are audits managed')
    assert payload.top_k == 4
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