
import chromadb
from pathlib import Path

# Location of our local vector database
DB_PATH = Path("data/chroma")

# Connect to persistent Chroma storage
client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

# List existing collections
collections = client.list_collections()

print("Available collections:")

for collection in collections:
    print("Name:", collection.name)
    print("Document chunks:", collection.count())

collection = client.get_collection("rag_documents")

records = collection.get(
    limit=3,
    include=["documents", "metadatas", "embeddings"]
)

for i, document_id in enumerate(records["ids"]):
    print("\nDocument ID:", document_id)
    print("Metadata:", records["metadatas"][i])
    print("Text:", records["documents"][i][:300])

    vector = records["embeddings"][i]

    print("Embedding dimensions:", len(vector))
    print("First 10 values:", vector[:10])
