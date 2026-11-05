# LocalLLM Support Copilot

LocalLLM Support Copilot is a full-stack application that hosts a local large language model with **vLLM** and connects it to a web-based customer-support workflow.

It demonstrates local LLM serving, OpenAI-compatible API integration, backend services, AI tool calling, and a modern web interface for managing and analyzing support tickets.

## What it does

The application is an AI-assisted customer-support copilot. Users can create and review support tickets in a Next.js web app, then use a locally hosted model to analyze tickets and provide assistance. Tickets, orders, and payments are stored in Supabase PostgreSQL, and the AI tool agent reads that verified business data through backend tools.

**Generate AI Reply** combines retrieved policy context with order/payment lookups to produce an editable reply draft.

```text
Next.js frontend (port 3000)
        |
        v
FastAPI backend (port 8080)
        |-- Ticket CRUD ----------------------> Supabase PostgreSQL
        |-- /analyze --> OpenAI SDK --> vLLM
        |
        `-- /tickets/{id}/ask
              |-- RAG retrieval <------------- Markdown knowledge base
              |     top-3 policy context
              v
            Tool agent <--> Order/payment tools <--> Supabase PostgreSQL
              |
              v
            OpenAI Python SDK
              |
              v
            vLLM (http://localhost:8000/v1)
              |
              v
            Qwen/Qwen3-8B-AWQ on NVIDIA RTX 4070 12 GB
```

## Components and roles

| Component | Role |
| --- | --- |
| Next.js | Web frontend for the ticket workflow (port 3000) |
| FastAPI | Application backend and HTTP API (port 8080) |
| Backend services | Ticket logic and the AI tool-agent loop |
| PolicyRetriever | In-memory Markdown retrieval using CPU embeddings and cosine similarity |
| Markdown knowledge base | Return, shipping, and troubleshooting guidance; separate from database records |
| Tools | Verified business-data retrieval for orders and payments |
| Supabase PostgreSQL | Persistent application data (tickets, orders, payments) |
| OpenAI Python SDK | Client library used to call the OpenAI-compatible vLLM API |
| vLLM | Local inference/model server that hosts the model (port 8000) |
| Qwen/Qwen3-8B-AWQ | Language model that writes the analysis and selects tool calls |
| NVIDIA RTX 4070 12 GB | Local GPU that runs the model |

## Highlights

- Hosts an LLM locally with vLLM in WSL2, using an NVIDIA RTX 4070 12 GB GPU.
- Connects FastAPI to vLLM through its OpenAI-compatible API.
- Provides a Next.js interface for creating, viewing, and analyzing support tickets.
- Persists tickets, orders, and payments in Supabase PostgreSQL.
- Demonstrates AI-agent tool calling for order and payment support workflows, where tool results are treated as verified backend data.
- Combines top-3 policy retrieval with the existing order/payment tool agent.
- Includes API and service tests, plus recorded vLLM latency benchmarks.

## Tech stack

- **Model:** Qwen/Qwen3-8B-AWQ
- **Model serving:** vLLM 0.27.1 inside WSL2 on an NVIDIA RTX 4070 12 GB
- **Backend:** Python, FastAPI, Pydantic, SQLAlchemy, OpenAI Python SDK
- **RAG:** SentenceTransformers, `BAAI/bge-small-en-v1.5`, NumPy cosine similarity; in-memory index
- **Data:** Supabase PostgreSQL
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Testing:** pytest

## Run locally

Start the local vLLM server first. The detailed model-serving setup is in [DEV.md](DEV.md).

The backend reads its PostgreSQL connection string from `backend/.env` (the `DATABASE_URL` variable), which points at the Supabase database. That file is local only and is gitignored.

Then run the backend from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --port 8080
```

Or activate the environment in CMD, then run the same Uvicorn command:

```cmd
.venv\Scripts\activate.bat
uvicorn backend.main:app --reload --port 8080
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend health endpoint is available at [http://localhost:8080/health](http://localhost:8080/health).

## Retrieval-Augmented Generation (RAG)

RAG grounds support replies in company documents covering returns, shipping, and troubleshooting. It complements structured order/payment lookups: policy text explains general rules, while backend tools verify customer-specific records. RAG is integrated into FastAPI's `POST /tickets/{ticket_id}/ask` and is accessible through **Generate AI Reply**.

```text
Knowledge base -> chunking -> embeddings -> cosine similarity -> top-3 context -> Qwen
```

The shared [PolicyRetriever](backend/services/rag_service.py) uses **BAAI/bge-small-en-v1.5** on CPU, reusing document embeddings in memory. The [knowledge base](backend/knowledge_base/) contains return and shipping policies plus a troubleshooting guide; retrieved passages supplement the existing agent without changing order/payment tools or `verified_facts`.

**Limitations:** The index is memory-only and does not automatically refresh after document edits. Retrieval has no minimum similarity threshold, so unsupported questions can still receive nearby passages. Policy context and prompt instructions do not guarantee factual answers or verified return eligibility; drafts require review.

## RAG testing and evaluation

Run from the project root with `.venv` active. Model-based commands need cached embedding weights or access to download them.

| Command | Purpose | vLLM / GPU requirement |
| --- | --- | --- |
| `python scripts/test_rag_answer.py` | Retrieve context and generate sample answers | Local vLLM/Qwen; GPU serving setup in DEV.md |
| `python scripts/evaluate_rag.py` | Evaluate standalone RAG responses | Local vLLM/Qwen; GPU serving setup in DEV.md |
| `python scripts/evaluate_rag.py --test-only` | Run offline evaluation helper tests | Neither; no models or network |
| `python -m pytest -v` | Run backend unit/API tests with mocked models | Neither; no real models or vLLM |

The evaluator uses [a small case dataset](scripts/rag_eval_cases.json) to report retrieval Hit@3 (whether the expected source filename appears in the top three chunks) and heuristic phrase matches in generated answers. Neither metric measures overall answer accuracy. It exercises the standalone RAG path rather than the full API/tool-agent workflow; backend tests separately cover retrieval, integration, fallback, and tool calling.

## Project structure

```text
backend/             FastAPI API, AI services, tools, and database layer
  knowledge_base/    Markdown support policies and troubleshooting guidance
frontend/            Next.js ticket workflow and editable AI reply drafts
benchmarks/          Recorded vLLM benchmark documentation and results
scripts/             RAG experiments/evaluation, tool checks, and benchmarks
supabase/            SQL migrations and Supabase configuration
tests/               Backend, RAG, lifecycle, API, and agent tests
DEV.md               Detailed Windows, WSL2, GPU, and vLLM setup notes
```

## vLLM benchmark results

The local model server was benchmarked directly, with no FastAPI, tool agent, or database in the path. `scripts/benchmark_vllm.py` posts to `http://localhost:8000/v1/chat/completions` using `Qwen/Qwen3-8B-AWQ`, the prompt `Explain what vLLM is in one sentence.`, and `max_tokens=64`.

| Workload | Total elapsed time | Successful requests |
| --- | ---: | ---: |
| 10 sequential requests | ~3.06 s average per request | 10 / 10 |
| 2 requests at the same time | 3.07 s | 2 / 2 |
| 4 requests at the same time | 3.37 s | 4 / 4 |
| 8 requests at the same time | 10.60 s | 8 / 8 |

These are client-observed latency measurements only. They show that wall-clock time stays close to the single-request baseline at low concurrency and grows clearly at concurrency 8. The benchmark does not measure GPU utilization or vLLM's internal scheduling, so these numbers do not by themselves prove GPU saturation or identify the exact bottleneck. Full measurements are recorded in [benchmarks/vllm_sequential_baseline.md](benchmarks/vllm_sequential_baseline.md).

## Development notes

The application expects vLLM at `http://localhost:8000/v1`, while the FastAPI backend runs on port `8080` so it does not conflict with the model server. The frontend runs on port `3000` and calls the backend at `http://localhost:8080`. See [DEV.md](DEV.md) for the full environment and vLLM setup guide, and [benchmarks/vllm_sequential_baseline.md](benchmarks/vllm_sequential_baseline.md) for the recorded benchmark results.

The vLLM benchmarks above measure direct generation requests and do not include RAG retrieval, FastAPI, or database work.
