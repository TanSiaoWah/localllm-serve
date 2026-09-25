# LocalLLM-Serve

## Project

Portfolio project: self-hosted AI customer-support copilot.

Architecture:

Next.js frontend
→ FastAPI backend
→ application/services
→ OpenAI Python SDK
→ vLLM OpenAI-compatible API
→ Qwen3-8B-AWQ
→ RTX 4070 12GB

## Important architecture rules

- `frontend/` = Next.js UI
- `backend/api/` = HTTP/API routes
- `backend/services/` = business and AI logic
- `backend/tools/` = functions available to the AI agent
- `backend/models.py` = Pydantic schemas
- `backend/data/` = current in-memory data
- vLLM provides the local inference server.
- OpenAI Python SDK is only the client used to communicate with vLLM.
- The LLM selects tools; Python executes tools.
- Tool results are treated as verified backend information.
- Prompt instructions are not security guarantees.
- Service-layer exceptions should not directly depend on HTTP.
- API routes translate service exceptions into HTTP responses.
- Tests should not require a running GPU/vLLM unless explicitly testing integration.

## Development rules

- Make tiny incremental changes.
- Do not implement multiple future features in one task.
- Preserve working functionality.
- Do not rewrite working code unnecessarily.
- Prefer simple, readable, interview-explainable code.
- Do not add database, authentication, Docker, RAG, Redis, Celery, Kubernetes, etc. unless specifically requested.
- Before modifying code, inspect the existing implementation.
- After changes, run the relevant tests/type checks.
- Do not modify unrelated files.

## Current important endpoints

GET /health

GET /tickets/
GET /tickets/{ticket_id}
POST /tickets/
PUT /tickets/{ticket_id}
DELETE /tickets/{ticket_id}

POST /tickets/{ticket_id}/analyze
POST /tickets/{ticket_id}/ask

## Current AI behavior

`/analyze` returns structured ticket analysis:
- issue
- category
- urgency

`/ask` returns:
- ticket_id
- answer
- verified_facts

The tool agent currently has:
- get_order_status
- get_payment_status
- maximum 5 turns

## Testing

Backend:

python -m pytest -v

Frontend:

cd frontend
npx tsc --noEmit