"""Order status lookup tool, backed by the database.

``get_order_status`` keeps its original name, argument, and response shape.
The ``orders`` dict below is retained only as the fixture source for the seed
script; lookups now go through the order repository.
"""

import logging

from backend.data import order_repository
from backend.data.database import SessionLocal

logger = logging.getLogger(__name__)

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
    logger.info("Order tool invoked: order_id=%s", order_id)

    if not order_id.startswith("ORD-"):
        logger.warning("Order tool rejected invalid identifier: order_id=%s", order_id)
        return {"error": "Invalid order ID"}

    try:
        session = SessionLocal()
        try:
            order = order_repository.get_order(session, order_id)
        finally:
            session.close()
    except Exception:
        # Return a controlled result and never leak database-specific details.
        logger.error("Order lookup failed: order_id=%s", order_id)
        return {"error": "Order service temporarily unavailable"}

    if order is None:
        logger.info("Order not found: order_id=%s", order_id)
        return {"error": "Order not found"}

    logger.info("Order found: order_id=%s", order_id)
    return {
        "product": order.product,
        "status": order.status,
        "payment_id": order.payment_id,
    }
