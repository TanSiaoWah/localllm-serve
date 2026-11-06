# LocalLLM Support Copilot

## Project Overview

A self-hosted AI customer-support copilot that combines local language-model
inference, policy retrieval, and backend tool execution. Support agents can
manage tickets, analyze issues, and generate editable response drafts informed
by company policies and order or payment records.

This AI Engineer portfolio project demonstrates how to connect a local LLM to
an application with persistent data, controlled tool access, and observable
request execution.

## Key Features

- **Ticket management:** Create, view, update, and delete support tickets stored in Supabase PostgreSQL.
- **Structured analysis:** Extract issue, category, and urgency through a dedicated AI endpoint with Pydantic output validation.
- **Response drafting:** Generate editable replies with retrieved policy context and a separate `verified_facts` list of executed tool results.
- **Read-only tools:** Look up order/payment status and search the external Kapruka product catalog. Kapruka order creation and tracking are blocked.

## Tech Stack

| Layer | Technologies |
|---|---|
| Frontend | Next.js, React, TypeScript, Tailwind CSS |
| Backend | Python 3.11, FastAPI, Pydantic, SQLAlchemy |
| Persistence | Supabase PostgreSQL |
| Local inference | Qwen3-8B-AWQ, vLLM, OpenAI Python SDK |
| Retrieval | Sentence Transformers, `BAAI/bge-small-en-v1.5`, NumPy, Markdown documents |
| External tools | MCP Python SDK, Kapruka MCP server |
| Delivery and validation | Docker Compose, GitHub Actions, pytest, TypeScript checks |

## System Architecture

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

The OpenAI Python SDK is the client for vLLM's OpenAI-compatible API. vLLM runs
separately in WSL2 on an NVIDIA RTX 4070 12 GB; it provides model inference.

## AI Request Workflow

1. **Request:** The Next.js interface sends a ticket question to FastAPI's `POST /tickets/{id}/ask` endpoint. FastAPI validates the input and loads the stored ticket.
2. **Retrieval:** The CPU embedding model retrieves the top three policy chunks from the [Markdown knowledge base](backend/knowledge_base), using cosine similarity against an in-memory index built at startup.
3. **Agent inference:** The [tool agent](backend/services/tool_agent_service.py) sends the ticket, question, and available policy context to Qwen through vLLM.
4. **Optional tool execution:** The LLM selects a tool; Python checks the allowed tool and arguments, executes the lookup, and returns its result to the model. The agent permits at most five LLM turns.
5. **Response:** FastAPI returns the generated answer and tool-derived facts. The frontend presents the draft for review and editing.

Retrieved policy passages and external catalog results are distinguished from
customer-specific backend facts. Retrieval and tool use provide evidence for
drafts; they do not guarantee factual correctness.

## Engineering Highlights

- **Tool control:** Request-specific tool selection and identifier checks constrain order/payment lookups; the [Kapruka adapter](backend/tools/kapruka_tools.py) exposes only product search.
- **Request protection:** Trimmed input limits cover ticket subjects, messages, and AI questions. Both LLM clients use explicit connection/read/write/pool timeouts with SDK retries disabled; API routes translate AI service errors into HTTP responses.
- **Observability:** [Correlated JSON traces](backend/services/agent_trace.py) record retrieval, LLM, and tool durations and outcomes without logging customer payloads.
- **RAG lifecycle:** Embeddings run on CPU and are reused across requests. Initialization failure allows degraded operation; `/health` reports liveness, while `/ready` reports RAG initialization status only.
- **Offline testing:** [Backend tests](tests) use mocked AI/MCP calls, fake embeddings, and isolated SQLite data. They do not require a GPU or live external services.
- **Evaluation:** [RAG evaluation](scripts/evaluate_rag.py), [agent evaluation](scripts/evaluate_qwen_agent.py), and a [vLLM benchmark report](benchmarks/vllm_sequential_baseline.md) examine retrieval, tool behavior, and inference latency. RAG answer checks use heuristic phrase matching rather than semantic accuracy scoring.
- **Containerization and CI:** [Docker Compose](compose.yaml) packages the frontend and backend with a persistent model cache; vLLM, Supabase, and Kapruka remain external. [GitHub Actions](.github/workflows/ci.yml) runs offline backend tests, Next.js type generation, and frontend TypeScript checks.
