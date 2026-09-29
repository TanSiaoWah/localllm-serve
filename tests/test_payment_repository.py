"""Focused tests for the payment repository.

Uses the same isolated in-memory SQLite approach as tests/test_seed.py, so
these tests do not connect to Supabase.
"""

from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.data.db_models import Base, Payment
from backend.data import payment_repository


def _make_engine():
    """Single-connection in-memory SQLite engine with a "support" schema attached."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _attach_support_schema(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("ATTACH DATABASE ':memory:' AS support")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


@pytest.fixture
def memory_session():
    """Provide an in-memory SQLite session seeded with the demo payment."""
    engine = _make_engine()
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    session.add(
        Payment(
            payment_id="PAY-88888",
            status="captured",
            amount=Decimal("1299.00"),
            currency="MYR",
        )
    )
    session.commit()
    yield session
    session.close()
    engine.dispose()


def test_get_existing_payment(memory_session):
    """An existing payment is returned with the expected field values."""
    payment = payment_repository.get_payment(memory_session, "PAY-88888")

    assert payment is not None
    assert payment.payment_id == "PAY-88888"
    assert payment.status == "captured"
    assert float(payment.amount) == 1299.00
    assert payment.currency == "MYR"


def test_get_missing_payment(memory_session):
    """A missing payment id returns None."""
    assert payment_repository.get_payment(memory_session, "PAY-NOPE") is None