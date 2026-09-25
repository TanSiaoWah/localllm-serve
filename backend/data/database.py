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
    {
        "id": 3,
        "customer_name": "David Tan",
        "subject": "Payment and order issue",
        "message": "I placed order ORD-12345 and I was charged RM1299, but I want to confirm whether my payment was successful and what the current status of my order is.",
        "status": "pending",
    },
]
