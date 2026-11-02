
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The team performs regular quality audits.",
    "The company conducts compliance reviews.",
    "The customer purchased a new laptop."
]

embeddings = model.encode(sentences)

print("Embeddings matrix shape:", embeddings.shape)
assert embeddings.shape == (3, 384)

for sentence, vector in zip(sentences, embeddings):
    print("\nSentence:", sentence)
    print("Dimensions:", len(vector))
    print("First 10 values:", vector[:10])


# Calculate the similarity between all three sentences
similarity_matrix = cosine_similarity(embeddings)

print("\nCosine Similarity Matrix:")
print(similarity_matrix)

print("\nIndividual comparisons:")

print(
    "Quality audits vs compliance reviews:",
    round(similarity_matrix[0][1], 4)
)

print(
    "Quality audits vs laptop purchase:",
    round(similarity_matrix[0][2], 4)
)

print(
    "Compliance reviews vs laptop purchase:",
    round(similarity_matrix[1][2], 4)
)

