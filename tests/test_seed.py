"""Focused tests for the seed data, without requiring the remote database.

Uses an in-memory SQLite database to exercise the idempotent seeding logic
in backend/data/seed.py.
"""

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.data import seed
from backend.data.db_models import Base, Order, Payment, Ticket


def _make_engine():
    """Return a single-connection in-memory SQLite engine with a "support" schema.

    The models are schema-qualified (support.*), which SQLite honours, so a
    database named "support" must be attached before create_all() and seeding
    run. StaticPool reuses one connection so the attached schema persists.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _attach_support_schema(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("ATTACH DATABASE ':memory:' AS support")
        # Enforce FKs so the tests catch a wrong insert order (e.g. seeding an
        # order before its payments row exists).
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    return engine


@pytest.fixture
def memory_session():
    """Provide a fresh in-memory SQLite session with the support schema created."""
    engine = _make_engine()
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()
    engine.dispose()


def test_ticket_seed_rows_preserve_exact_demo_values():
    """Ticket rows hold the exact demo values (id, name, subject, message, status)."""
    rows = seed.ticket_seed_rows()
    assert len(rows) == 3

    by_id = {row["id"]: row for row in rows}
    assert by_id[1] == {
        "id": 1,
        "customer_name": "Alice Smith",
        "subject": "Login issue",
        "message": "I cannot log into my account.",
        "status": "open",
    }
    assert by_id[2] == {
        "id": 2,
        "customer_name": "Bob Jones",
        "subject": "Billing question",
        "message": "Why was I charged twice this month?",
        "status": "pending",
    }
    assert by_id[3] == {
        "id": 3,
        "customer_name": "David Tan",
        "subject": "Payment and order issue",
        "message": (
            "I placed order ORD-12345 and I was charged RM1299, but I want to "
            "confirm whether my payment was successful and what the current "
            "status of my order is."
        ),
        "status": "pending",
    }


def test_payment_seed_rows_match_fixture():
    """Payment row matches the PAY-88888 fixture."""
    rows = seed.payment_seed_rows()
    assert len(rows) == 1
    assert rows[0]["payment_id"] == "PAY-88888"
    assert rows[0]["status"] == "captured"
    assert rows[0]["currency"] == "MYR"
    assert float(rows[0]["amount"]) == 1299.00


def test_order_seed_rows_match_fixture():
    """Order row matches the ORD-12345 fixture."""
    rows = seed.order_seed_rows()
    assert len(rows) == 1
    assert rows[0] == {
        "order_id": "ORD-12345",
        "product": "Laptop",
        "status": "shipped",
        "payment_id": "PAY-88888",
    }


def test_seed_inserts_all_records_once(memory_session):
    """First run inserts everything; a second run inserts nothing."""
    first = seed.seed(memory_session)
    assert first == 5  # 3 tickets + 1 order + 1 payment

    second = seed.seed(memory_session)
    assert second == 0

    assert memory_session.query(Ticket).count() == 3
    assert memory_session.query(Order).count() == 1
    assert memory_session.query(Payment).count() == 1


def test_seed_is_idempotent_across_sessions(memory_session):
    """New sessions do not create duplicates either."""
    seed.seed(memory_session)
    seed.seed(memory_session)

    assert memory_session.query(Ticket).count() == 3
    assert memory_session.query(Order).count() == 1
    assert memory_session.query(Payment).count() == 1


def test_sequence_sync_is_noop_on_sqlite(memory_session):
    """The PostgreSQL-only sequence sync is a no-op on SQLite, so seed works.

    This exercises the dialect guard: if the sync tried to run its
    PostgreSQL ``setval`` SQL against SQLite it would fail. Verifying the
    actual sequence advancement requires PostgreSQL, which the SQLite test
    harness cannot provide, so that behavior is not asserted here.
    """
    assert memory_session.get_bind().dialect.name != "postgresql"
    assert seed.seed(memory_session) == 5


def test_ticket_id_sequence_sql_targets_max_plus_one():
    """The sync SQL advances support.tickets.id via setval / MAX(id)."""
    sql = str(seed.TICKET_ID_SEQUENCE_SQL)
    assert "setval" in sql
    assert "pg_get_serial_sequence('support.tickets', 'id')" in sql
    assert "MAX(id)" in sql