"""Payment status lookup tool, backed by the database.

``get_payment_status`` keeps its original name, argument, and response shape.
Data is retrieved through the payment repository; there is no hard-coded data
in this module.
"""

import logging

from backend.data import payment_repository
from backend.data.database import SessionLocal

logger = logging.getLogger(__name__)


def get_payment_status(payment_id: str):
    """Return the payment details for a given payment ID from the database."""
    logger.info("Payment tool invoked: payment_id=%s", payment_id)

    if not payment_id.startswith("PAY-"):
        logger.warning(
            "Payment tool rejected invalid identifier: payment_id=%s", payment_id
        )
        return {"error": "Invalid payment ID"}

    try:
        session = SessionLocal()
        try:
            payment = payment_repository.get_payment(session, payment_id)
        finally:
            session.close()
    except Exception:
        # Return a controlled result and never leak database-specific details.
        logger.error("Payment lookup failed: payment_id=%s", payment_id)
        return {"error": "Payment service temporarily unavailable"}

    if payment is None:
        logger.info("Payment not found: payment_id=%s", payment_id)
        return {"error": "Payment not found"}

    logger.info("Payment found: payment_id=%s", payment_id)
    return {
        "status": payment.status,
        "amount": float(payment.amount),
        "currency": payment.currency,
    }
