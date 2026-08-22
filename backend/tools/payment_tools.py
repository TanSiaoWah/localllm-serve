# Fake payment data
payments = {
    "PAY-88888": {
        "status": "captured",
        "amount": 1299.00,
        "currency": "MYR",
    },
}


# Normal Python function that looks up a payment
def get_payment_status(payment_id: str):
    """Return the payment details for a given payment ID."""
    if not payment_id.startswith("PAY-"):
        return {"error": "Invalid payment ID"}
    return payments.get(payment_id, {"error": "Payment not found"})
