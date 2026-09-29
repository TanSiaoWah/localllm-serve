"""Data-access layer for orders.

This repository is the only place that performs SQLAlchemy database access
for orders. ``get_order`` returns the ``Order`` ORM object (or ``None`` if the
row is missing) so the service layer can decide what that means. It takes an
existing SQLAlchemy ``Session`` and never creates one, never commits, and
contains no HTTP/FastAPI logic.
"""

from sqlalchemy.orm import Session

from backend.data.db_models import Order


def get_order(session: Session, order_id: str):
    """Return a single order by its id, or ``None`` if it does not exist."""
    return session.get(Order, order_id)