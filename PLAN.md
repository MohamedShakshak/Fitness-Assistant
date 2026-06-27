# Fitness RAG Project Plan

## Overview
End-to-end RAG system for a fitness exercise Q&A chatbot. Users ask exercise-related questions and get sourced answers from a knowledge base of 4,387 exercises. Personal/educational use. Python stack with uv.

## Tech Stack
- **Language:** Python 3.12
- **Package Manager:** uv
- **RAG Framework:** LangChain (LCEL chains, ChatPromptTemplate, HuggingFaceEmbeddings)
- **Embedding Model:** `BAAI/bge-base-en-v1.5` (768 dims, local, sentence-transformers)
- **Reranker:** `BAAI/bge-reranker-base`
- **Vector Store:** Qdrant (Docker, hybrid search: dense + BM25)
- **LLM:** OpenRouter (default, `liquid/lfm-2.5-1.2b-instruct:free`) + Ollama (fallback, local)
- **Config:** `.env` + `.env.example`, python-dotenv, Pydantic Settings
- **Backend:** FastAPI (REST endpoints)
- **Frontend:** Streamlit (chat UI with sidebar filters + LLM toggle)

## Architecture Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Use case | Exercise Q&A chatbot | Not workout generator, not form checker |
| 2 | Exercise chunking | 1 chunk per exercise (labeled sections format) | Each exercise is a discrete knowledge unit |
| 3 | Empty exercise handling | Drop exercises with no description AND no instructions | 27% of chunks were empty noise; ~3,204 exercises remain |
| 4 | Chunk text format | Labeled sections with field headers, numbered instructions, omit empty fields | BM25 gets distinct tokens per field; dense embeddings capture structure |
| 5 | Chunk data class | LangChain `Document` (`page_content` + `metadata`) | Direct LCEL compatibility, no manual conversion needed |
| 6 | Guide chunking | REMOVED — Fitness Wiki dropped (only 1 inert index page) | Future guide content will use LangChain `RecursiveCharacterTextSplitter` |
| 7 | Embedding model | `BAAI/bge-base-en-v1.5` (768d, 512 ctx, local) | MTEB ~64, better semantic quality, handles longer chunks |
| 8 | Vector store | Qdrant (Docker) | Hybrid search support, production-like, scalable |
| 9 | Metadata storage | All fields in Qdrant payload for filtering; same fields in `page_content` for retrieval | Redundancy is intentional — metadata for exact filtering, text for semantic retrieval |
| 10 | Default retrieval | Top 5 | Balance between coverage and relevance |
| 11 | Search type | Hybrid (semantic dense + BM25 sparse) | Handles natural language AND exact keyword queries |
| 12 | Reranking | Yes — `BAAI/bge-reranker-base` | Better relevance ordering, trained on multilingual data |
| 13 | LLM default | OpenRouter (`nvidia/nemotron-3-super-120b-a12b:free`) | Free tier, better quality than previous model |
| 14 | LLM fallback | Ollama Qwen3.5:9B (local) | Free, private, no internet needed |
| 15 | LLM switching | Configurable at runtime via `.env` | Users can switch providers in Streamlit sidebar |
| 16 | System prompt | Strict fitness expert persona + guardrails + medical disclaimer | Handles Phase 5 guardrails in prompt |
| 17 | Citation/source attribution | Yes — cite sources from retrieved docs | Builds trust, verifiable answers |
| 18 | Conversation memory | 5-turn sliding window | Sufficient for fitness Q&A |
| 19 | Application interface | FastAPI backend + Streamlit frontend (separate processes) | Clean separation, backend reusable |
| 20 | API/Frontend comms | REST endpoints over HTTP | Backend reusable for other clients |
| 21 | API design | `POST /query`, `POST /search`, `GET /filters`, `GET /health` | Covers core RAG + search + filter needs |
| 22 | Qdrant runtime | Docker container on localhost:6333 | Production-like, persistent |
| 23 | Config approach | `.env` + `.env.example` with Pydantic Settings | All runtime values (especially secrets) come from `.env`; `config.py` defines schema, types, and defaults |
| 24 | Config validation | `OPENROUTER_API_KEY` required when `LLM_PROVIDER=openrouter` | Fails fast instead of silently using blank key |
| 25 | API schemas | Single source of truth in `schemas.py`, imported by `main.py` | No duplication; typed schemas (`ChatMessage`, `SourceInfo`) |
| 26 | Secrets in code | No hardcoded API keys; all loaded from `.env` via `settings` | `.env` is gitignored; scripts use `settings.OPENROUTER_API_KEY` |
| 27 | Chunk IDs | Slug-based (`ex_3_4_sit_up_1`) | Deterministic, debuggable, no collisions across re-indexes |
| 28 | `chunk_type` metadata | `"exercise"` (kept for forward-compatibility with future guide content) | Near-zero cost, enables type-based filtering later |
| 29 | `has_description`/`has_instructions` metadata | Kept — enables filtering exercises with instructions | Useful for queries like "show me exercises with step-by-step instructions" |
| 30 | `source` metadata | List format (`["wrkout", "kaggle_gym"]`) | Preserves multi-source provenance, enables array filtering in Qdrant |
| 31 | `url` metadata | Omitted for exercises (mostly empty in datasets) | Can be added back if/when datasets include URLs |
| 32 | Metadata deserialization | JSON-serialized lists deserialized on read from Qdrant | Lists stored as JSON strings in payload; `_deserialize_payload()` converts them back |
| 33 | Multi-value filters | Use Qdrant `MatchAny` for list values | Streamlit sends lists for multi-select; `MatchValue` only matches single strings |
| 34 | Chat history flow | Send history *before* adding current message to session_state | Prevents current question from appearing twice in LLM context |
| 35 | API error handling | try/except with `HTTPException` on all endpoints | No raw Python tracebacks leaked to clients; proper 400/500 responses |
| 36 | CORS | `CORSMiddleware` with `allow_origins=["*"]` | Enables browser-based clients and Swagger UI |
| 37 | Collection creation | Check existence before delete; only recreate when needed | Prevents accidental data loss on restart |
| 38 | Logging | Python `logging` module with configurable `LOG_LEVEL` in `.env` | Structured log levels; no `print()` statements in production code |
| 39 | Configurable settings | `API_URL`, `EMBEDDING_DEVICE`, `SPARSE_MODEL`, `LLM_MAX_RETRIES`, `LLM_TIMEOUT`, `LLM_MAX_TOKENS`, `LOG_LEVEL`, `EVAL_MODEL`, `EVAL_BATCH_SIZE` | No hardcoded values; everything configurable via `.env` |
| 40 | Evaluation approach | Lean hybrid: custom deterministic metrics + RAGAS Faithfulness only | Skipped noisy metrics (AnswerCorrectness, ContextPrecision/Recall, ResponseRelevancy); manual 25 Q&A pairs for reliable ground truth |
| 41 | Eval naming matching | Normalized string matching (lowercase, strip trailing punctuation, collapse whitespace) | Fixes false negatives from trailing dashes in exercise names (e.g., "Dumbbell bench press-" vs "Dumbbell Bench Press") |
| 42 | Docker deployment | `docker-compose.yml` with Qdrant + FastAPI + Streamlit services | Single `docker compose up` for full stack; Qdrant healthcheck gates API startup |

---

## Phases

### Phase 1: Data Collection & Preparation ✅ COMPLETED
- [x] Download wrkout/exercises.json (873 exercises)
- [x] Download Kaggle Gym Exercise Dataset (2,918 exercises)
- [x] Download Kaggle Fitness Exercises Dataset (1,324 exercises)
- [x] ~~Scrape The Fitness Wiki~~ ( Removed — only 1 inert index page, no exercises)
- [x] Unify, clean, deduplicate into single schema — **4,387 unique exercises**
- [ ] MET Compendium, Wikipedia, Wikidata (deferred — scripts exist)

### Phase 2: Chunking, Embedding & Vector Store ✅ COMPLETED (v2)
- [x] Restructure project modules (config, chunking, embedding, vectorstore, retrieval, generation, api, app)
- [x] Add dependencies: qdrant-client, sentence-transformers, fastapi, streamlit, httpx, langchain, etc.
- [x] Switch from custom `Chunk` dataclass to `langchain.schema.Document` for LCEL compatibility
- [x] ExerciseChunker: 1 chunk per exercise, labeled-section format with field headers
- [x] Drop 1,183 empty exercises (no description AND no instructions) — ~3,204 indexed
- [x] ~~GuideChunker~~ — removed (Fitness Wiki dropped)
- [x] Omit empty fields from chunk text (no "Force:" line when force is empty)
- [x] Number multi-step instructions; single-step as sentence
- [x] Embeddings: `BAAI/bge-base-en-v1.5` (768d, 512 ctx)
- [x] Qdrant: Docker container, `fitness_exercises` collection, ~3,204 points
- [x] Hybrid search: dense vectors + BM25 sparse vectors
- [x] Reranker: `BAAI/bge-reranker-base`
- [x] All metadata fields stored as Qdrant payload for filtering

### Phase 3: Retrieval Pipeline ✅ COMPLETED
- [x] Hybrid search (dense + BM25) via Qdrant `query_points` with `Prefetch`
- [x] Reranking with cross-encoder after retrieval
- [x] Top 5 results by default, configurable
- [x] Metadata filtering (body_part, equipment, level, category)

### Phase 4: LLM Generation Layer ✅ COMPLETED
- [x] System prompt: fitness-expert persona with guardrails + medical disclaimer
- [x] LangChain ChatPromptTemplate with `MessagesPlaceholder` for chat history
- [x] Context injection from retrieved docs with source citations
- [x] OpenRouter (cloud) + Ollama (local) LLM providers via LangChain
- [x] 5-turn sliding window for chat history

### Phase 5: Guardrails & Safety ✅ COMPLETED (in system prompt)
- [x] Fitness-only validation (handled in system prompt)
- [x] Medical disclaimer on all responses
- [x] Off-topic query refusal

### Phase 6: Application Layer ✅ COMPLETED
- [x] FastAPI backend: `/query`, `/search`, `/filters`, `/health`
- [x] CORS middleware enabled on FastAPI
- [x] API error handling with try/except and HTTPException (400/500)
- [x] Streamlit frontend: chat UI with sidebar (LLM provider toggle, metadata filters)
- [x] Conversation memory (5-turn sliding window)
- [x] Chat history duplication bug fixed (send history before current message)
- [x] Streamlit API_URL configurable via environment variable

### Phase 7: Production Hardening 🔲 IN PROGRESS

#### Critical Bugs Fixed ✅
- [x] Source metadata deserialization bug — JSON-stringified lists (`source`, `primary_muscles`, `secondary_muscles`) now properly deserialized when reading from Qdrant
- [x] Multi-select filter bug — filters with multiple values now use `MatchAny` instead of broken `MatchValue(str(list))`
- [x] Chat history duplication — current user question appeared twice in LLM context; now sent only once
- [x] Destructive `recreate_collection` — now checks if collection exists first, deletes only if needed
- [x] Duplicate import in qdrant_store.py removed
- [x] Bare `except Exception` in Streamlit — replaced with specific `httpx.ConnectError`/`TimeoutException`

#### Engineering Quality ✅ COMPLETED
- [x] **pytest test suite** — 79 tests, all passing (tests/test_chunker.py, test_prompts.py, test_schemas.py, test_qdrant_store.py, test_eval_metrics.py, test_config.py, test_api.py)
- [x] Add `logging` module throughout, remove `print()` statements
- [x] Configurable settings: `API_URL`, `EMBEDDING_DEVICE`, `SPARSE_MODEL`, `LLM_MAX_RETRIES`, `LLM_TIMEOUT`, `LOG_LEVEL`, `LLM_MAX_TOKENS`
- [x] `.gitignore` cleanup (evals/experiments/, temp files, eval output files)
- [x] Remove placeholder `main.py`, update `pyproject.toml` description
- [x] `PydanticDeprecatedSince20` fix: `m.dict()` → `m.model_dump()` in API
- [x] `src/__init__.py` was a directory — fixed to proper file
- [ ] ~Thread-safe singletons (`_embedder`, `_reranker`, `_qdrant_client`) with `threading.Lock`~
- [ ] ~Type annotations: `Optional[int]` instead of `int = None`~
- [ ] ~Docstrings on all public functions~
- [ ] ~`SearchResponse.results` typed as `list[SearchResult]` instead of `list[dict]`~
- [ ] ~Empty `__init__.py` files with `__all__` exports~
- [ ] ~Input validation on API endpoints (max query length, rate limiting)~

### Phase 8: Deployment ✅ COMPLETED
- [x] Docker Compose (Qdrant + FastAPI + Streamlit) — `docker-compose.yml` + `Dockerfile`
- [ ] Deploy locally or to cloud
- [ ] Monitor retrieval hit rates, LLM costs, user feedback

### Phase 9: Evaluation 🔲 IN PROGRESS

#### Eval Dataset ✅ COMPLETED
- [x] 25 fitness Q&A pairs across 5 categories (muscle_equipment, how_to, muscle_targeting, level_based, comparison) — `data/eval/qa_pairs.json`
- [x] 5 off-topic guardrail questions (medical, nutrition, general) — `data/eval/off_topic_questions.json`
- [x] All expected exercise names verified against chunk DB

#### Retrieval Metrics ✅ COMPLETED (no reranker — model download issue)
- [x] Deterministic Recall@K, MRR, Hit Rate — `scripts/eval_retrieval.py`
- [x] Name normalization (lowercase, strip trailing punctuation, collapse whitespace) — fixes false negatives from trailing dashes
- [x] Per-category breakdown + JSON output to `evals/experiments/`
- [x] Results: Recall@5=0.3867, MRR=0.38, Hit Rate=0.56 (see `evals/experiments/summary.md`)

#### RAGAS + Guardrail Metrics 🔲 BLOCKED (OpenRouter rate limit)
- [x] `src/evaluation/ragas_setup.py` — `get_ragas_llm()` factory wrapping OpenRouter (max_tokens=4096)
- [x] `src/evaluation/metrics.py` — Custom DiscreteMetrics: `disclaimer_metric`, `citation_metric`, `refusal_metric`
- [x] `scripts/run_evaluation.py` — Full orchestrator: generation + retrieval + RAGAS Faithfulness + guardrails
- [x] RAGAS 0.4.3 import compatibility fixed (vertexai shim file)
- [x] DiscreteMetric API fixed: `metric.ascore(llm=llm, response=answer)` (not `single_turn_ascore`)
- [x] Faithfulness context fix: passing full chunk text (not just exercise names)
- [ ] **Full eval run** — blocked on OpenRouter 50 free req/day limit; rerun after reset

---

## Data Sources

### ✅ Downloaded & Processed

| Source | Records | Key Fields | Status |
|--------|---------|------------|--------|
| wrkout/exercises.json | 873 | name, muscles, equipment, type, mechanics, instructions, level, force | ✅ |
| Kaggle: Gym Exercise Dataset | 2,918 | title, description, type, body part, equipment, level, rating | ✅ |
| Kaggle: Fitness Exercises Dataset | 1,324 | name, body part, equipment, target muscle, secondary muscles, instructions | ✅ |
| ~~The Fitness Wiki~~ | ~~14 pages~~ | ~~title, URL, content~~ | ❌ Removed (only inert index page) |
| **Unified Dataset** | **4,387 exercises** | name, description, instructions, category, body_part, muscles, equipment, level, force, mechanic, source | ✅ |
| **Qdrant Index** | **~3,204 points** | Dense (768d) + BM25 sparse vectors, all metadata (empties dropped) | ✅ |

### 🔲 Deferred (scripts exist, can download later)

| Source | What You Get | Status |
|--------|-------------|--------|
| MET Compendium | 1,000+ activities with MET calorie values | 🔲 Manual download |
| Wikipedia exercise articles | 43 articles on exercises, muscles, techniques | 🔲 Script ready (needs User-Agent fix) |
| Wikidata SPARQL | Structured exercise-muscle-equipment relationships | 🔲 Script ready |
| ExRx.net | Exercise directory, kinesiology, workout templates | 🔲 Not yet scraped |
| OpenStax A&P Textbook | Musculoskeletal anatomy chapters | 🔲 Manual download |
| WHO/HHS Guidelines PDFs | Activity guidelines for all populations | 🔲 Manual download |

### Out of Scope (skipped)
- USDA FoodData Central, Open Food Facts, Nutrition APIs
- Diet/meal planning sources
- Any supplement/nutrition-only source

---

## Project Structure

```
fitness-rag/
├── data/
│   ├── raw/                              # Downloaded originals
│   │   ├── wrkout_exercises/             # 873 exercises (JSON + git repo)
│   │   ├── kaggle_gym_dataset/           # 2,918 exercises (CSV)
│   │   └── kaggle_fitness_exercises/     # 1,324 exercises (CSV)
│   ├── processed/                        # Unified, deduplicated
│   │   └── exercises.json               # 4,387 unified exercises
│   ├── chunks/                           # RAG-ready chunks
│   │   └── exercises/exercise_chunks.json # ~3,204 exercise chunks (LangChain Documents)
│   └── eval/                             # Evaluation datasets
│       ├── qa_pairs.json                # 25 fitness Q&A pairs (5 categories)
│       └── off_topic_questions.json     # 5 off-topic guardrail questions
├── src/
│   ├── __init__.py
│   ├── config.py                         # Pydantic Settings, .env loader, API key validation
│   ├── pipeline.py                       # End-to-end RAG (LangChain LCEL)
│   ├── chunking/
│   │   └── chunker.py                   # ExerciseChunker (LangChain Document format)
│   ├── embedding/
│   │   ├── embedder.py                  # LangChain HuggingFaceEmbeddings (bge-base-en-v1.5)
│   │   └── reranker.py                  # Cross-encoder reranker (bge-reranker-base)
│   ├── vectorstore/
│   │   └── qdrant_store.py              # Qdrant hybrid search (dense + BM25)
│   ├── retrieval/
│   │   └── retriever.py                  # embed → hybrid search → rerank
│   ├── generation/
│   │   ├── llm_client.py                # ChatOllama + ChatOpenAI (OpenRouter)
│   │   └── prompts.py                   # Fitness expert system prompt
│   ├── data_collection/                  # Phase 1 download & scrape scripts
│   ├── data_processing/                  # Phase 1 unify script
│   ├── evaluation/
│   │   ├── ragas_setup.py               # RAGAS judge LLM factory (OpenRouter)
│   │   └── metrics.py                    # Custom DiscreteMetrics (disclaimer, citation, refusal)
│   ├── api/
│   │   ├── main.py                       # FastAPI endpoints
│   │   └── schemas.py                   # Pydantic request/response models
│   └── app/
│       └── streamlit_app.py              # Streamlit chat UI
├── scripts/
│   ├── index_data.py                     # One-command: load → chunk → embed → upsert
│   ├── eval_retrieval.py                 # Deterministic retrieval metrics (no LLM)
│   ├── run_evaluation.py                 # Full eval: retrieval + RAGAS + guardrails
│   ├── test_search.py                    # Test retrieval only
│   ├── test_models.py                    # Test free OpenRouter models
│   └── list_free_models.py              # List free models on OpenRouter
├── evals/
│   └── experiments/                      # Eval results (gitignored)
│       ├── retrieval_results.json        # Deterministic retrieval metrics
│       └── summary.md                    # Human-readable summary
├── tests/                                # pytest test suite (79 tests)
│   ├── test_chunker.py                  # ExerciseChunker unit tests
│   ├── test_prompts.py                  # Prompt template tests
│   ├── test_schemas.py                  # API Pydantic schema tests
│   ├── test_qdrant_store.py             # Serialization & filter building tests
│   ├── test_eval_metrics.py             # Retrieval metric & name normalization tests
│   ├── test_config.py                   # Settings defaults tests
│   └── test_api.py                      # FastAPI endpoint integration tests (mocked)
├── docker-compose.yml                    # Qdrant + FastAPI + Streamlit
├── Dockerfile                            # Python 3.12-slim + uv
├── .env                                  # Secrets (gitignored)
├── .env.example                          # Config template (committed)
├── pyproject.toml                         # uv project config + pytest config + dev deps
├── PLAN.md                               # This file
├── README.md                             # Project documentation
└── .gitignore
```

---

## Chunking Strategy

### Exercise chunks (1 per exercise, labeled-section format)
Each exercise record is converted to a structured document with field headers:

```
Exercise: 3/4 Sit-Up
Category: Strength (Compound)
Target: Abdominals
Secondary: Hip flexors, lower back
Equipment: Body only
Level: Beginner
Force: Pull

The 3/4 sit-up is a bodyweight exercise focused on the muscles of the core..."

How to perform:
1. Lie down on the floor and secure your feet...
2. Place your hands behind or to the side of your head...
3. Flex your hips and spine to raise your torso...
```

**Rules:**
- Empty fields are omitted entirely (no "Force:" line if force is empty)
- Category combines category + mechanic: `"Strength (Compound)"`
- Multi-step instructions are numbered; single-step is a sentence
- Blank lines separate header section, description, and instructions
- Exercises with no description AND no instructions are **dropped entirely**
- Text goes in LangChain `Document.page_content`; metadata goes in `Document.metadata`

### Guide chunks (future)
When guide/text content is added later, will use LangChain `RecursiveCharacterTextSplitter`.

---

## Configuration (.env)

```env
# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=fitness_exercises

# Embedding
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIM=768
EMBEDDING_DEVICE=cpu

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-base

# Sparse Model
SPARSE_MODEL=Qdrant/bm25

# LLM - Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b

# LLM - OpenRouter (default)
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
LLM_MAX_RETRIES=3
LLM_TIMEOUT=120
LLM_MAX_TOKENS=2048

# LLM Provider: "ollama" or "openrouter"
LLM_PROVIDER=openrouter

# Retrieval
TOP_K=5
RERANK_TOP_K=5

# Application
API_URL=http://localhost:8000
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Eval
EVAL_MODEL=nvidia/nemotron-3-super-120b-a12b:free
EVAL_BATCH_SIZE=5

# Chat
MAX_HISTORY_TURNS=5
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/query` | Main RAG query. Body: `{query, chat_history?, filters?, llm_provider?, top_k?}` → `{answer, sources, context_used}` |
| `POST` | `/search` | Retrieval-only (no LLM). Body: `{query, filters?, top_k?}` → `{results: [{id, text, score, metadata}]}` |
| `GET` | `/filters` | Returns available filter options (unique body_parts, equipment, levels, categories) |
| `GET` | `/health` | Health check |

---

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total unique exercises (unified) | 4,387 |
| Indexed exercises (after dropping empties) | ~3,204 |
| With instructions | 2,074 (47%) |
| With descriptions | 1,359 (31%) |
| Dropped (no description AND no instructions) | 1,183 (27%) |
| Categories | Strength, Stretching, Plyometrics, Cardio, Powerlifting, Olympic Weightlifting, Strongman |
| Qdrant points | ~3,204 |
| Embedding dim | 768 |
| Sparse vectors | BM25 (Qdrant/bm25 via fastembed) |

---

## How to Run

### Prerequisites
1. **Docker** running (for Qdrant)
2. **Ollama** (optional, for local LLM)
3. **OpenRouter API key** (for cloud LLM)

### Start Qdrant
```bash
docker run -d -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

### Index Data
```bash
uv run python scripts/index_data.py
```

### Start API
```bash
uv run python -m src.api.main
```

### Start Streamlit
```bash
uv run streamlit run src/app/streamlit_app.py
```

### Run Tests
```bash
uv run pytest tests/ -v                 # All 79 tests
uv run pytest tests/test_api.py -v      # API tests only
```

### Run Evaluation
```bash
uv run python scripts/eval_retrieval.py --no-rerank    # Deterministic retrieval metrics (no LLM)
uv run python scripts/run_evaluation.py --no-rerank    # Full eval (retrieval + RAGAS + guardrails)
```

### Docker Compose (Full Stack)
```bash
docker compose up -d                   # Qdrant + FastAPI + Streamlit
docker compose logs -f                  # View logs
docker compose down                     # Stop all services
```

---

## Next Steps

1. ~~All Phase 1-6 tasks~~ ✅
2. ~~Chunking v2~~ — switch to LangChain `Document`, labeled-section format, drop empty exercises ✅
3. ~~Embedding & Reranker upgrade~~ — `BAAI/bge-base-en-v1.5` (768d) + `BAAI/bge-reranker-base` ✅
4. ~~Critical bug fixes~~ — deserialization, filters, chat duplication, CORS, error handling ✅
5. ~~Production hardening~~ — logging, tests, .gitignore, pyproject.toml, model_dump fix ✅
6. ~~Docker Compose~~ — containerize Qdrant + FastAPI + Streamlit together ✅
7. ~~Retrieval evaluation~~ — 25 Q&A pairs, Recall@K, MRR, Hit Rate, name normalization ✅
8. **Full evaluation run** — RAGAS Faithfulness + guardrail metrics (blocked on OpenRouter rate limit; rerun after reset)
9. **Fix reranker download** — `bge-reranker-base` model stalls at ~268MB; retry or manual download
10. **Add guide content** — when/if Wikipedia or a proper Fitness Wiki scrape is done, use LangChain `RecursiveCharacterTextSplitter`