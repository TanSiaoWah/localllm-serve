from fastapi import FastAPI

app = FastAPI()


# In-memory list to store tickets (no database yet)
tickets = [
    {
        "id": 1,
        "customer_name": "Alice Smith",
        "subject": "Login issue",
        "message": "I cannot log into my account.",
        "status": "open",
    },
    {
        "id": 2,
        "customer_name": "Bob Jones",
        "subject": "Billing question",
        "message": "Why was I charged twice this month?",
        "status": "pending",
    },
]


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/tickets")
def get_tickets():
    """Return the full list of tickets."""
    return tickets
