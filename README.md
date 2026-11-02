# RAG Quality Lab

A hands-on Retrieval-Augmented Generation (RAG) application with FastAPI, local Chroma, sentence-transformers embeddings, optional Claude answer generation, and pytest.

## Architecture

`TXT/MD/PDF -> text extraction -> overlapping chunks -> embeddings -> Chroma`

`question -> embedding -> Chroma top-k search -> similarity cutoff -> evidence -> extractive result OR Claude -> answer + sources`

**Important:** Claude does not generate embeddings. `all-MiniLM-L6-v2` embeds passages locally. The first run downloads model files and requires internet access.

## Windows PowerShell setup

```powershell
cd rag-quality-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
uvicorn app.main:app --reload
```

Python 3.11 or 3.12 is recommended. Open `http://127.0.0.1:8000/docs` for the interactive API interface.

### First manual workflow

1. `GET /health`: confirm status `ok`.
2. `POST /documents`: upload `sample_docs/team_quality_policy.txt` as `file`.
3. `POST /retrieve`: `{"question":"When is the quality review meeting?", "top_k":4}`. Examine source and similarity.
4. `POST /ask`: same payload. Default `extractive` mode returns the best passage with `[1]`, without needing any API key.
5. Run `python -m evals.run_eval` from repository root to compare expected retrieval and abstention behavior.

### Optional Claude mode

In `.env` put `ANSWER_PROVIDER=claude`, `ANTHROPIC_API_KEY=<your key>`, and a model ID accessible to your account in `ANTHROPIC_MODEL`. You must load `.env` yourself with your shell, or use `python -m dotenv run -- uvicorn app.main:app --reload`. Never commit secrets. API calls incur costs. Validate your model ID from your Claude API console.

## Code tour

- `app/core.py`: pure text normalization, overlapping chunks, input validation, evidence prompt, answer abstention. Start here.
- `app/service.py`: converts files to text, computes vector embeddings, persists in Chroma, retrieves closest passages, optionally calls Claude.
- `app/main.py`: REST API endpoints, request validation, and HTTP error mapping.
- `tests/test_core.py`: deterministic tests for chunking, validation, unanswerable questions, citation range, and prompt assembly.
- `tests/test_api.py`: FastAPI HTTP integration with a fake service: no downloads, no keys.
- `evals/golden.jsonl`: tiny labeled Q&A dataset for retrieval evaluation.
- `evals/run_eval.py`: runs the configured live backend against labeled examples.

## Practical limits and follow-up work

- This is a **learning MVP**, not an enterprise-secure multi-user application. Do not upload private resumes or customer files to a public deployment.
- Default extractive mode returns a passage rather than a synthesized answer. Claude mode requires a paid API key.
- Similarity cutoff is a heuristic, not proof of relevance; tune against labeled answerable/unanswerable queries. Poorly tuned cutoff can cause irrelevant answers or false abstentions.
- Prompt instructions alone cannot guarantee resistance to prompt injection. Use retrieval tests, attack cases, and human review.
- The live evaluation script has a tiny sample and does **not** independently prove groundedness or factual accuracy. Improve it with span/citation verification and a human-labeled evaluation set.
- Image-only PDFs are not OCR'd. PDFs with embedded text work.
- Production upgrades: authentication, document isolation, delete/replace support, metadata filters, malware scanning, token-aware chunks, rate limits, telemetry, concurrency safe storage, and a server-backed Chroma instance for Railway deployment. Chroma PersistentClient is designed primarily for local development, not ephemeral production containers.
