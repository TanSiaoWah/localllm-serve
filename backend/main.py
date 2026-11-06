import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api.tickets import router as tickets_router
from backend.services.agent_trace import configure_trace_logging
from backend.services.rag_service import PolicyRetriever

logger = logging.getLogger(__name__)


def create_policy_retriever():
    """Construct the optional CPU retriever only when application startup asks."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    kb_dir = Path(__file__).resolve().parent / "knowledge_base"
    return PolicyRetriever(model, kb_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_trace_logging()
    app.state.policy_retriever = None
    try:
        retriever = create_policy_retriever()
        retriever.initialize()
    except Exception:
        logger.exception("Policy retrieval initialization failed; continuing without RAG")
    else:
        app.state.policy_retriever = retriever
        logger.info("Policy retriever initialized on CPU")

    try:
        yield
    finally:
        app.state.policy_retriever = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    """Report RAG startup status without performing initialization or I/O."""
    retriever = getattr(request.app.state, "policy_retriever", None)
    return {"status": "ready" if retriever is not None else "degraded"}


app.include_router(tickets_router)
