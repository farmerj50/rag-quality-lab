"""Local embeddings + persisted Chroma store + optional Claude generation."""
import hashlib
import io
import os
from pathlib import Path

from .core import ABSTAIN, build_messages, chunk_text, extractive_answer, validate_question


class RagService:
    def __init__(self, client=None, embedder=None, provider=None):
        import chromadb
        self.embedder = embedder or self._local_embedder()
        self.client = client or chromadb.PersistentClient(path=os.getenv('CHROMA_PATH', './data/chroma'))
        self.collection = self.client.get_or_create_collection(name='rag_documents', metadata={'hnsw:space': 'cosine'})
        self.provider = provider or os.getenv('ANSWER_PROVIDER', 'extractive')
        self.minimum_similarity = float(os.getenv('MIN_SIMILARITY', '0.25'))

    @staticmethod
    def _local_embedder():
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'))
        return lambda texts: model.encode(texts, normalize_embeddings=True).tolist()

    def ingest(self, filename: str, raw: bytes) -> dict:
        suffix = Path(filename).suffix.lower()
        if len(raw) > 5 * 1024 * 1024:
            raise ValueError('Maximum file size is 5 MB')
        if suffix in {'.txt', '.md'}:
            text = raw.decode('utf-8')
        elif suffix == '.pdf':
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(raw))
            if len(reader.pages) > 100:
                raise ValueError('Maximum PDF length is 100 pages')
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        else:
            raise ValueError('Supported extensions: .txt, .md, .pdf')
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError('No extractable text found')
        digest = hashlib.sha256(raw).hexdigest()[:16]
        ids = [f'{digest}:{i}' for i in range(len(chunks))]
        vectors = self.embedder(chunks)
        self.collection.upsert(ids=ids, documents=chunks, embeddings=vectors,
                               metadatas=[{'source': Path(filename).name, 'part': i, 'file_id': digest}
                                          for i in range(len(chunks))])
        return {'filename': Path(filename).name, 'chunks': len(chunks), 'file_id': digest}

    def retrieve(self, question: str, top_k: int = 4) -> list[dict]:
        question = validate_question(question)
        if not 1 <= top_k <= 10:
            raise ValueError('top_k must be between 1 and 10')
        if self.collection.count() == 0:
            return []
        result = self.collection.query(query_embeddings=self.embedder([question]),
                                       n_results=min(top_k, self.collection.count()),
                                       include=['documents', 'metadatas', 'distances'])
        sources = []
        for text, meta, distance in zip(result['documents'][0], result['metadatas'][0], result['distances'][0]):
            similarity = 1.0 - float(distance)
            if similarity >= self.minimum_similarity:
                sources.append({'text': text, 'source': meta['source'], 'part': meta['part'],
                                'similarity': round(similarity, 4)})
        return sources

    def ask(self, question: str, top_k: int = 4) -> dict:
        sources = self.retrieve(question, top_k)
        if not sources:
            return {'answer': ABSTAIN, 'sources': [], 'provider': self.provider}
        if self.provider == 'extractive':
            answer = extractive_answer(sources)
        elif self.provider == 'claude':
            import anthropic
            system, user = build_messages(question, sources)
            response = anthropic.Anthropic().messages.create(
                model=os.environ['ANTHROPIC_MODEL'], max_tokens=500,
                system=system, messages=[{'role': 'user', 'content': user}])
            answer = ''.join(block.text for block in response.content if block.type == 'text')
        else:
            raise ValueError('ANSWER_PROVIDER must be extractive or claude')
        return {'answer': answer, 'sources': sources, 'provider': self.provider}
