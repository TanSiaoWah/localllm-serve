# Fake order data
orders = {
    "ORD-12345": {
        "product": "Laptop",
        "status": "shipped",
        "payment_id": "PAY-88888",
    },
}


# Normal Python function that looks up an order
def get_order_status(order_id: str):
    """Return the order details for a given order ID."""
    if not order_id.startswith("ORD-"):
        return {"error": "Invalid order ID"}
    return orders.get(order_id, {"error": "Order not found"})
