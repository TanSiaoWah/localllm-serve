# LocalLLM Support Copilot

A self-hosted customer-support copilot built with Next.js and FastAPI. It uses **Qwen3-8B-AWQ through local vLLM** to analyze tickets and generate editable replies, with Supabase PostgreSQL storing tickets, orders, and payments.

## Features

- **Ticket management:** Create, view, update, delete, and analyze support tickets.
- **Local inference:** The OpenAI Python SDK connects to vLLM running in WSL2 on an NVIDIA RTX 4070 12 GB.
- **Tool calling:** The model selects order/payment lookups; Python executes authorized tools and returns `verified_facts` alongside the answer.
- **RAG:** Retrieved company policies provide context for support replies.
- **Kapruka MCP:** Read-only product search for explicit Kapruka catalog questions; order creation and tracking are blocked. See the [adapter](backend/tools/kapruka_tools.py).
- **Evaluation and observability:** Offline tests, model evaluation scripts, and correlated JSON traces cover agent behavior and execution timing.

## Architecture

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
              |-- Product search <-----------> External Kapruka MCP server
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

The [agent service](backend/services/tool_agent_service.py) controls tool access for each request.

## Run locally

Follow [DEV.md](DEV.md) for the environment and vLLM setup. Start vLLM at `http://localhost:8000/v1`, and set `DATABASE_URL` in the gitignored `backend/.env` to your Supabase PostgreSQL connection string.

With the environment ready, start the backend from the project root:

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

Open [localhost:3000](http://localhost:3000), select a ticket, and use **Generate AI Reply**. Review the draft before using it. Check the backend at [GET /health](http://localhost:8080/health).

## Run with Docker

Docker Compose runs FastAPI on port 8080 and Next.js on port 3000. vLLM stays
in WSL2; Supabase PostgreSQL and Kapruka MCP remain external. Neither container
requires GPU access.

Before starting, run Docker Desktop with Linux containers enabled and configure
`DATABASE_URL` in the gitignored `backend/.env` (see `backend/.env.example`).
Credentials are supplied at runtime, not included in images. Run these commands
from the repository root:

```cmd
docker compose up --build -d
docker compose ps
docker compose logs --tail 100 backend frontend
```

Check the [frontend](http://localhost:3000) and
[backend health endpoint](http://localhost:8080/health). Ports are published only
on this computer.

The backend reaches host-side vLLM through `host.docker.internal:8000`; Windows/WSL2
forwarding and firewall settings must allow access. First startup downloads and
initializes the CPU RAG model, which may delay readiness. Model files persist in
a Docker volume; inspect backend logs for initialization errors. A healthy API
does not confirm external integrations or RAG are working.

Stop containers while retaining the model cache:

```cmd
docker compose down
```

## Evaluation and observability

- [RAG evaluation](scripts/evaluate_rag.py) ([sample cases](scripts/rag_eval_cases.json)) and [agent evaluation](scripts/evaluate_qwen_agent.py) exercise model behavior using local services.
- [Agent traces](backend/services/agent_trace.py) correlate retrieval, LLM, and tool timings in terminal JSON logs without customer payloads.
- See the [benchmark report](benchmarks/vllm_sequential_baseline.md) for direct-vLLM latency measurements.

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

## Testing

With the virtual environment active, run the offline backend suite from the project root:

```powershell
python -m pytest tests -q
```

Tests use mocked external services and an isolated database; no GPU, vLLM, live MCP, or real Supabase credentials are required. For frontend type checking, run `npx tsc --noEmit` from `frontend/`.
