# LocalLLM Support Copilot

LocalLLM Support Copilot is a full-stack application that hosts a local large language model with **vLLM** and connects it to a web-based customer-support workflow.

It demonstrates local LLM serving, OpenAI-compatible API integration, backend services, AI tool calling, and a modern web interface for managing and analyzing support tickets.

## What it does

The application is an AI-assisted customer-support copilot. Users can create and review support tickets in a Next.js web app, then use a locally hosted model to analyze tickets and provide assistance. Tickets, orders, and payments are stored in Supabase PostgreSQL, and the AI tool agent reads that verified business data through backend tools.

```text
Browser / Next.js frontend (port 3000)
        |
        v
FastAPI backend (port 8080)
        |
        +--> Ticket data -------------------+
        |                                   |
        +--> Tool agent (order and payment  |
                tools) ---------------------+--> Supabase PostgreSQL
        |
        v
OpenAI Python SDK
        |
        v
vLLM OpenAI-compatible server (http://localhost:8000/v1)
        |
        v
Qwen/Qwen3-8B-AWQ (local language model)
        |
        v
NVIDIA RTX 4070 12 GB
```

## Components and roles

| Component | Role |
| --- | --- |
| Next.js | Web frontend for the ticket workflow (port 3000) |
| FastAPI | Application backend and HTTP API (port 8080) |
| Backend services | Ticket logic and the AI tool-agent loop |
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
- Includes API and service tests, plus recorded vLLM latency benchmarks.

## Tech stack

- **Model:** Qwen/Qwen3-8B-AWQ
- **Model serving:** vLLM 0.27.1 inside WSL2 on an NVIDIA RTX 4070 12 GB
- **Backend:** Python, FastAPI, Pydantic, SQLAlchemy, OpenAI Python SDK
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
use CMD:

```cmd
.venv\Scripts\activate.bat
```

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The backend health endpoint is available at [http://localhost:8080/health](http://localhost:8080/health).

## Project structure

```text
backend/       FastAPI API, AI services, tool definitions, and database layer
frontend/      Next.js customer-support interface
benchmarks/    Recorded vLLM benchmark documentation and results
scripts/       Small scripts for validating vLLM, tool calling, and benchmarking
supabase/      SQL migrations and Supabase configuration
tests/         Backend, AI-service, and agent-evaluation tests
DEV.md         Detailed Windows, WSL2, GPU, and vLLM setup notes
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
