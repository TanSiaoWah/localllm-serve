"""Focused tests for the order repository.

Uses the same isolated in-memory SQLite approach as tests/test_seed.py, so
these tests do not connect to Supabase.
"""

from decimal import Decimal

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.data.db_models import Base, Order, Payment
from backend.data import order_repository


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
    """Provide an in-memory SQLite session seeded with the demo order + payment."""
    engine = _make_engine()
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    # Payment first so the order's foreign key resolves; flush it so the
    # payment row exists before the order INSERT is emitted.
    session.add(
        Payment(
            payment_id="PAY-88888",
            status="captured",
            amount=Decimal("1299.00"),
            currency="MYR",
        )
    )
    session.flush()
    session.add(
        Order(
            order_id="ORD-12345",
            product="Laptop",
            status="shipped",
            payment_id="PAY-88888",
        )
    )
    session.commit()
    yield session
    session.close()
    engine.dispose()


def test_get_existing_order(memory_session):
    """An existing order is returned with the expected field values."""
    order = order_repository.get_order(memory_session, "ORD-12345")

    assert order is not None
    assert order.order_id == "ORD-12345"
    assert order.product == "Laptop"
    assert order.status == "shipped"
    assert order.payment_id == "PAY-88888"


def test_get_missing_order(memory_session):
    """A missing order id returns None."""
    assert order_repository.get_order(memory_session, "ORD-NOPE") is None