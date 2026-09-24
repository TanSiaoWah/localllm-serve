# LocalLLM Support Copilot

LocalLLM Support Copilot is a full-stack application that hosts a local large language model with **vLLM** and connects it to a web-based customer-support workflow.

It demonstrates local LLM serving, OpenAI-compatible API integration, backend services, AI tool calling, and a modern web interface for managing and analyzing support tickets.

## What it does

The application is an AI-assisted customer-support copilot. Users can create and review support tickets in a Next.js web app, then use a locally hosted model to analyze tickets and provide assistance.

```text
Next.js frontend (port 3000)
        |
        v
FastAPI backend (port 8080)
        |
        v
Local vLLM OpenAI-compatible server (port 8000)
        |
        v
Local LLM running on the GPU
```

## Highlights

- Hosts an LLM locally with vLLM in WSL2, using an NVIDIA GPU.
- Connects FastAPI to vLLM through its OpenAI-compatible API.
- Provides a Next.js interface for creating, viewing, and analyzing support tickets.
- Demonstrates AI-agent tool calling for order and payment support workflows.
- Includes API and service tests.

## Tech stack

- **Model serving:** vLLM, WSL2, NVIDIA GPU
- **Backend:** Python, FastAPI, Pydantic, OpenAI Python SDK
- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Testing:** pytest

## Run locally

Start the local vLLM server first. The detailed model-serving setup is in [DEV.md](DEV.md).

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
backend/       FastAPI API, AI services, tool definitions, and data layer
frontend/      Next.js customer-support interface
scripts/       Small scripts for validating vLLM and tool calling
tests/         Backend and AI-service tests
DEV.md         Detailed Windows, WSL2, GPU, and vLLM setup notes
```

## Development notes

The application expects vLLM at `http://localhost:8000/v1`, while the FastAPI backend runs on port `8080` so it does not conflict with the model server. See [DEV.md](DEV.md) for the full environment and vLLM setup guide.
