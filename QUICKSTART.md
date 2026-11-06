# LocalLLM-Serve Quick Reference

Assumes dependencies and `backend/.env` are already configured. Choose native
or Docker startup; both use ports 3000 and 8080, so do not run both together.
vLLM runs separately in WSL2 and is not managed by Docker Compose.

## vLLM — WSL2

Open WSL2 from Windows CMD:

```cmd
wsl
```

In WSL2 Bash, activate the existing environment documented in [DEV.md](DEV.md)
and start the model:

```bash
cd ~/vllm-test
source .venv/bin/activate
vllm serve Qwen/Qwen3-8B-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3
```

Keep this terminal running. Press **Ctrl+C** to stop vLLM.

## Native startup — Windows CMD

Backend terminal:

```cmd
.venv\Scripts\activate.bat
uvicorn backend.main:app --reload --port 8080
```

Frontend terminal:

```cmd
cd frontend
npm run dev
```

Press **Ctrl+C** in each terminal to stop its server.

## Docker Compose — Windows CMD

With Docker Desktop running Linux containers, use the repository root:

Start existing images, or rebuild and start after changes:

```cmd
docker compose up -d
docker compose up --build -d # Rebuild and startup
```

Inspect status and follow logs (**Ctrl+C** exits log streaming only):

```cmd
docker compose ps
docker compose logs --tail 100 -f backend frontend
```

Stop containers while retaining the RAG model cache:

```cmd
docker compose down
```

Compose reaches vLLM through `host.docker.internal:8000`. First-time RAG model
download and initialization can delay backend readiness; inspect backend logs.

## URLs and quick checks — Windows CMD

- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API docs: [http://localhost:8080/docs](http://localhost:8080/docs)
- vLLM API: [http://localhost:8000/v1](http://localhost:8000/v1)

```cmd
curl.exe -I http://localhost:3000
curl.exe http://localhost:8080/health
curl.exe http://localhost:8080/ready
curl.exe http://localhost:8000/v1/models
```

`/health` returns `{"status":"ok"}` for liveness. `/ready` returns HTTP 200 with
`{"status":"ready"}` or `{"status":"degraded"}` for RAG initialization only;
neither endpoint verifies external services.