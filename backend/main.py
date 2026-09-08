from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.tickets import router as tickets_router

app = FastAPI()

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


app.include_router(tickets_router)
