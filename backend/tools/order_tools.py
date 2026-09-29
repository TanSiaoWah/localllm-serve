"""Order status lookup tool, backed by the database.

``get_order_status`` keeps its original name, argument, and response shape.
The ``orders`` dict below is retained only as the fixture source for the seed
script; lookups now go through the order repository.
"""

from backend.data import order_repository
from backend.data.database import SessionLocal

# Fixture data for the seed script (not used for lookups in this module).
orders = {
    "ORD-12345": {
        "product": "Laptop",
        "status": "shipped",
        "payment_id": "PAY-88888",
    },
}


def get_order_status(order_id: str):
    """Return the order details for a given order ID from the database."""
    if not order_id.startswith("ORD-"):
        return {"error": "Invalid order ID"}

    session = SessionLocal()
    try:
        order = order_repository.get_order(session, order_id)
    finally:
        session.close()

    if order is None:
        return {"error": "Order not found"}

    return {
        "product": order.product,
        "status": order.status,
        "payment_id": order.payment_id,
    }
