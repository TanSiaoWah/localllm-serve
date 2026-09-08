# Backend — How to Start the Server

The backend is a FastAPI application. Its entry point is `backend/main.py`.

## What runs when you start it

- A FastAPI app is created in `backend/main.py`
- CORS is enabled for the frontend at `http://localhost:3000`
- The tickets API router is registered (see `backend/api/tickets.py`)
- A health check endpoint is available at `GET /health`

## Prerequisites

- Python 3.12+
- A virtual environment at the project root (`.venv/`)
- The following packages installed in that environment:

```text
fastapi
uvicorn
pydantic
openai
```

Note: there is currently no `requirements.txt` in the project. Install the
packages above into the project's `.venv` if they are missing.

## Steps to start the server

### 1. Open a terminal at the project root

The app must be started from the project root (the folder that contains the
`backend/` directory), because the code uses imports like
`from backend.api.tickets import ...`.

```powershell
cd D:\Github_Website\localllm-serve
```

### 2. Activate the virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD:

```cmd
.venv\Scripts\activate.bat
```

### 3. Start the server with uvicorn

```powershell
uvicorn backend.main:app --reload --port 8080
```

- `backend.main:app` means: the `app` object inside `backend/main.py`
- `--reload` restarts the server automatically when code changes
- `--port 8080` runs the backend on port **8080**
- The server starts at: `http://127.0.0.1:8080`

> **Port note:** the backend uses port **8080** because port **8000** is
> reserved for the vLLM server (see `backend/services/ai_service.py`, which
> connects to `http://localhost:8000/v1`). The frontend already expects the
> backend at `http://localhost:8080`.

## How to verify it is running

Open this URL in a browser (or use `curl` / an API tool):

```text
http://127.0.0.1:8080/health
```

Expected response:

```json
{"status": "ok"}
```

You can also open the interactive API docs:

```text
http://127.0.0.1:8080/docs
```

## Stopping the server

Press `Ctrl+C` in the terminal where uvicorn is running.
