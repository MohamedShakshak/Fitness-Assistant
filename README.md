# FitAssist — Fitness Exercise RAG Assistant

End-to-end RAG system for fitness exercise Q&A. Ask questions like *"What exercises target my chest?"*, *"How do I deadlift?"*, or *"Give me beginner leg exercises"* and get sourced, expert-level answers from a knowledge base of **3,204 exercises**.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.18-red?logo=qdrant&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.137-teal?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit&logoColor=white)

---

## Features

- **3,204 exercises** from 3 datasets, unified and deduplicated (1,183 empty exercises dropped)
- **Hybrid search** — dense embeddings (BGE-base-en-v1.5, 768d) + BM25 sparse vectors
- **Cross-encoder reranking** (BGE-reranker-base) for improved relevance
- **Qdrant** vector database with metadata filtering (body part, equipment, level, category)
- **LangChain** RAG pipeline with fitness-expert system prompt
- **OpenRouter** (cloud) + **Ollama** (local) LLM support — switch at runtime
- **FastAPI** backend + **Streamlit** frontend
- **Source citations** — every answer references its source datasets
- **Medical disclaimer** built into every response
- **Evaluation framework** — deterministic retrieval metrics (Recall@K, MRR, Hit Rate) + RAGAS Faithfulness + custom guardrail metrics
- **Test suite** — 79 pytest tests covering chunking, prompts, schemas, Qdrant store, retrieval metrics, config, and API endpoints
- **Docker Compose** — one command for the full stack (Qdrant + FastAPI + Streamlit)

---

## Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────┐
│  Streamlit    │────▶│  FastAPI      │
│  Chat UI      │     │  /query       │
└──────────────┘     └──────┬───────┘
                             │
                    ┌────────▼────────┐
                    │  RAG Pipeline    │
                    │  (LangChain LCEL) │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
┌──────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
        │  Embedder   │ │  Qdrant   │ │  Reranker  │
        │  (768d)     │ │  Hybrid    │ │  (BGE)     │
        └────────────┘ │  Search    │ └───────────┘
                      └─────┬──────┘
                            │
                     ┌──────▼──────┐
                     │  LLM         │
                     │  (OpenRouter │
                     │   / Ollama)  │
                     └─────────────┘
```

---

## Quick Start

### 1. Prerequisites

- **Python 3.12+**
- **Docker** (for Qdrant)
- **uv** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **OpenRouter API key** ([get one free](https://openrouter.ai/keys)) or **Ollama** ([install](https://ollama.ai))

### 2. Clone & Install

```bash
git clone <your-repo-url>
cd fitness-rag
uv sync
```

### 3. Configure

Copy `.env.example` to `.env` and set your keys:

```bash
cp .env.example .env
# Edit .env with your OpenRouter API key
```

### 4. Start Qdrant

```bash
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 5. Index Data

```bash
uv run python scripts/index_data.py
```

This downloads the embedding model (first run ~430MB), chunks data, embeds, and upserts everything to Qdrant (~3-5 minutes on first run).

### 6. Run the App

```bash
# Terminal 1 — API
uv run python -m src.api.main

# Terminal 2 — Streamlit UI
uv run streamlit run src/app/streamlit_app.py
```

Open http://localhost:8501 to start chatting.

### 7. Docker Compose (Optional — Full Stack)

```bash
docker compose up -d
```

This starts Qdrant, the FastAPI backend, and Streamlit frontend in one command.

---

## Evaluation

### Retrieval Metrics (Deterministic — No LLM Required)

```bash
uv run python scripts/eval_retrieval.py --no-rerank
```

Measures Recall@K, MRR, and Hit Rate against 25 manually curated Q&A pairs across 5 categories:

| Category | Recall@5 | MRR | Hit Rate |
|----------|----------|-----|----------|
| muscle_equipment | 0.13 | 0.20 | 0.40 |
| how_to | 0.80 | 0.47 | 0.80 |
| muscle_targeting | 0.10 | 0.07 | 0.20 |
| level_based | 0.30 | 0.40 | 0.40 |
| comparison | 0.60 | 0.77 | 1.00 |
| **Overall** | **0.39** | **0.38** | **0.56** |

### Full Evaluation (RAGAS Faithfulness + Guardrails)

```bash
uv run python scripts/run_evaluation.py --no-rerank
```

Adds RAGAS Faithfulness, citation rate, disclaimer rate, and off-topic refusal rate. Requires an OpenRouter API key with available rate limit.

### Test Suite

```bash
uv run pytest tests/ -v
```

79 tests covering chunking, prompts, schemas, Qdrant store serialization, retrieval metrics, configuration, and API endpoints.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Main RAG query → `{answer, sources, context_used}` |
| `POST` | `/search` | Retrieval-only (no LLM) → `{results: [{id, text, score, metadata}]}` |
| `GET` | `/filters` | Available filter values (body_part, equipment, level, category) |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are some beginner chest exercises with dumbbells?"}'
```

---

## Data Sources

| Source | Exercises | License |
|--------|-----------|---------|
| [wrkout/exercises.json](https://github.com/wrkout/exercises.json) | 873 | Unlicense (public domain) |
| [Kaggle: Gym Exercise Dataset](https://www.kaggle.com/datasets/niharika41298/gym-exercise-data) | 2,918 | CC0-1.0 |
| [Kaggle: Fitness Exercises Dataset](https://www.kaggle.com/datasets/omarxadel/fitness-exercises-dataset) | 1,324 | MIT |
| **Unified** | **4,387** (3,204 indexed) | — |

---

## Project Structure

```
fitness-rag/
├── data/
│   ├── raw/                    # Downloaded originals
│   ├── processed/              # Unified exercises.json
│   ├── chunks/                 # RAG-ready chunks (LangChain Documents)
│   └── eval/                   # 25 Q&A pairs + 5 off-topic questions
├── src/
│   ├── config.py               # Pydantic Settings, .env loader, API key validation
│   ├── pipeline.py             # End-to-end RAG pipeline
│   ├── chunking/chunker.py     # Exercise chunking (LangChain Document format)
│   ├── embedding/
│   │   ├── embedder.py         # BGE-base-en-v1.5 (768d, LangChain)
│   │   └── reranker.py         # BGE-reranker-base
│   ├── vectorstore/qdrant_store.py  # Qdrant hybrid search + serialization
│   ├── retrieval/retriever.py  # Search + rerank pipeline
│   ├── generation/
│   │   ├── llm_client.py       # ChatOllama + ChatOpenAI
│   │   └── prompts.py          # System prompt + templates
│   ├── evaluation/
│   │   ├── ragas_setup.py      # RAGAS judge LLM factory
│   │   └── metrics.py          # Custom guardrail metrics
│   ├── api/
│   │   ├── main.py             # FastAPI endpoints
│   │   └── schemas.py          # Pydantic models
│   └── app/streamlit_app.py    # Streamlit chat UI
├── scripts/
│   ├── index_data.py           # One-command indexing
│   ├── eval_retrieval.py       # Deterministic retrieval metrics
│   ├── run_evaluation.py       # Full eval (RAGAS + guardrails)
│   ├── test_search.py          # Test retrieval
│   └── test_models.py          # Test OpenRouter models
├── tests/                      # 79 pytest tests (unit + integration)
├── evals/experiments/          # Eval results
├── docker-compose.yml          # Qdrant + FastAPI + Streamlit
├── Dockerfile                  # Python 3.12-slim + uv
├── .env                        # Secrets (gitignored)
├── .env.example                # Config template
├── pyproject.toml               # Dependencies + pytest config + dev deps
├── PLAN.md                     # Detailed project plan (42 architecture decisions)
└── README.md                   # This file
```

---

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `QDRANT_COLLECTION` | `fitness_exercises` | Qdrant collection name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Sentence-transformers model |
| `EMBEDDING_DIM` | `768` | Embedding dimensions |
| `EMBEDDING_DEVICE` | `cpu` | Device for embedding (`cpu` or `cuda`) |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder reranker model |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse embedding model for BM25 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Ollama model name |
| `OPENROUTER_API_KEY` | — | Your OpenRouter API key |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API URL |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter model |
| `LLM_PROVIDER` | `openrouter` | `"openrouter"` or `"ollama"` |
| `LLM_MAX_RETRIES` | `3` | LLM request retries |
| `LLM_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `LLM_MAX_TOKENS` | `2048` | Max generation tokens |
| `TOP_K` | `5` | Number of results to retrieve |
| `RERANK_TOP_K` | `5` | Number of results after reranking |
| `API_URL` | `http://localhost:8000` | FastAPI URL (used by Streamlit) |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MAX_HISTORY_TURNS` | `5` | Chat history window |
| `EVAL_MODEL` | `nvidia/nemotron-3-super-120b-a12b:free` | RAGAS judge LLM model |
| `EVAL_BATCH_SIZE` | `5` | Evaluation batch size |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| Package Manager | uv |
| RAG Framework | LangChain (LCEL) |
| Embedding | sentence-transformers `BAAI/bge-base-en-v1.5` (768d, local) |
| Reranker | `BAAI/bge-reranker-base` |
| Vector Store | Qdrant (Docker, hybrid: dense + BM25) |
| LLM (cloud) | OpenRouter (configurable models) |
| LLM (local) | Ollama (Qwen3.5:9B) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Config | Pydantic Settings + `.env` |

---

## Available OpenRouter Models (Free Tier)

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `liquid/lfm-2.5-1.2b-instruct:free` | 1.2B | Fast | Basic |
| `meta-llama/llama-3.2-3b-instruct:free` | 3B | Medium | Good |
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | Slow | Excellent |
| `nvidia/nemotron-3-ultra-550b-a55b:free` | 550B | Very slow | Best |

Free models are rate-limited. For production, consider a paid model like `gpt-4o-mini`.

---

## License

This project is for personal/educational use. Exercise data is sourced from publicly available datasets (CC0, MIT, Unlicense licenses). See individual data sources for specific license terms.

---

## Acknowledgments

- [wrkout/exercises.json](https://github.com/wrkout/exercises.json) — Exercise database (Unlicense)
- [Kaggle Gym Exercise Dataset](https://www.kaggle.com/datasets/niharika41298/gym-exercise-data) — Exercise data (CC0)
- [Kaggle Fitness Exercises Dataset](https://www.kaggle.com/datasets/omarxadel/fitness-exercises-dataset) — Exercise data (MIT)
- [Qdrant](https://qdrant.tech/) — Vector search engine
- [LangChain](https://langchain.com/) — RAG framework
- [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) — Embedding model
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) — Reranker model