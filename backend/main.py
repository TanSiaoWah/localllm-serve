from fastapi import FastAPI

from backend.api.tickets import router as tickets_router

app = FastAPI()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


app.include_router(tickets_router)
