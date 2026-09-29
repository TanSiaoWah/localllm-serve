"""Data-access layer for payments.

This repository is the only place that performs SQLAlchemy database access
for payments. ``get_payment`` returns the ``Payment`` ORM object (or ``None``
if the row is missing) so the service layer can decide what that means. It
takes an existing SQLAlchemy ``Session`` and never creates one, never commits,
and contains no HTTP/FastAPI logic.
"""

from sqlalchemy.orm import Session

from backend.data.db_models import Payment


def get_payment(session: Session, payment_id: str):
    """Return a single payment by its id, or ``None`` if it does not exist."""
    return session.get(Payment, payment_id)