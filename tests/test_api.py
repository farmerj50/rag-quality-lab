from fastapi.testclient import TestClient
from app.main import app, get_service
from app.core import ABSTAIN


class FakeService:
    def ingest(self, filename, raw):
        if not filename.endswith('.txt'):
            raise ValueError('Supported extensions: .txt, .md, .pdf')
        return {'filename': filename, 'chunks': 1, 'file_id': 'test'}

    def retrieve(self, question, top_k):
        return [{'text': 'The warranty is 12 months.', 'source': 'policy.txt',
                 'part': 0, 'similarity': 0.91}] if 'warranty' in question.lower() else []

    def ask(self, question, top_k):
        sources = self.retrieve(question, top_k)
        return {'answer': 'The warranty is 12 months. [1]' if sources else ABSTAIN,
                'sources': sources, 'provider': 'extractive'}


def test_api_end_to_end_with_fake_service():
    app.dependency_overrides[get_service] = lambda: FakeService()
    try:
        with TestClient(app) as client:
            assert client.get('/health').json() == {'status': 'ok'}
            r = client.post('/documents', files={'file': ('policy.txt', b'The warranty is 12 months.')})
            assert r.status_code == 200
            r = client.post('/retrieve', json={'question': 'What is the warranty?'})
            assert r.json()['sources'][0]['source'] == 'policy.txt'
            r = client.post('/ask', json={'question': 'What is the warranty?'})
            assert '[1]' in r.json()['answer']
            r = client.post('/ask', json={'question': 'Who won the game?'})
            assert r.json()['answer'] == ABSTAIN
            assert client.post('/ask', json={'question': ''}).status_code == 422
    finally:
        app.dependency_overrides.clear()


def test_bad_upload():
    app.dependency_overrides[get_service] = lambda: FakeService()
    try:
        with TestClient(app) as client:
            r = client.post('/documents', files={'file': ('bad.exe', b'x')})
            assert r.status_code == 400
    finally:
        app.dependency_overrides.clear()
