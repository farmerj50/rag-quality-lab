"""HTTP endpoints. Start: uvicorn app.main:app --reload"""
from functools import lru_cache
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .service import RagService

app = FastAPI(title='RAG Quality Lab', version='0.1.0')


@lru_cache
def get_service():
    return RagService()


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=10)


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'application': 'RAG Quality Lab',
        'message': 'API is running syccessfully'
        }


@app.post('/documents')
async def ingest(file: UploadFile = File(...), service: RagService = Depends(get_service)):
    try:
        # Cap streaming read rather than trusting Content-Length.
        raw = await file.read(5 * 1024 * 1024 + 1)
        return service.ingest(file.filename or 'unknown', raw)
    except (UnicodeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/retrieve')
def retrieve(payload: Question, service: RagService = Depends(get_service)):
    try:
        return {'sources': service.retrieve(payload.question, payload.top_k)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/ask')
def ask(payload: Question, service: RagService = Depends(get_service)):
    try:
        return service.ask(payload.question, payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
