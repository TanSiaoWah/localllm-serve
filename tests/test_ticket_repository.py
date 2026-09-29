"""Focused tests for the ticket repository.

Uses the same isolated in-memory SQLite approach as tests/test_seed.py, so
these tests do not connect to Supabase.
"""

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.data.db_models import Ticket
from backend.data import ticket_repository


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
    """Provide a fresh in-memory SQLite session with the tickets table created.

    The tickets table is created with an ``INTEGER PRIMARY KEY`` so SQLite can
    auto-generate ids. (``BIGINT`` is not a rowid alias in SQLite, so it cannot
    auto-increment; PostgreSQL's ``bigint identity`` is unaffected.)
    """
    engine = _make_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE support.tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_name TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    message TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        )
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.close()
    engine.dispose()


def _create_ticket(session, customer_name="Alice Smith", subject="Login issue"):
    """Create a ticket via the repository and commit it."""
    ticket = ticket_repository.create_ticket(
        session,
        customer_name=customer_name,
        subject=subject,
        message="I cannot log into my account.",
    )
    session.commit()
    return ticket


def test_list_tickets_empty(memory_session):
    """An empty table returns an empty list."""
    assert ticket_repository.list_tickets(memory_session) == []


def test_list_tickets_returns_created_tickets(memory_session):
    """Created tickets appear in the list, ordered by id."""
    a = _create_ticket(memory_session, customer_name="Alice Smith")
    b = _create_ticket(memory_session, customer_name="Bob Jones")

    result = ticket_repository.list_tickets(memory_session)

    assert [t.id for t in result] == [a.id, b.id]


def test_get_ticket_existing(memory_session):
    """A stored ticket can be retrieved by id."""
    created = _create_ticket(memory_session)

    fetched = ticket_repository.get_ticket(memory_session, created.id)

    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.customer_name == "Alice Smith"
    assert fetched.subject == "Login issue"


def test_get_ticket_missing(memory_session):
    """A missing id returns None (no 404 raised here)."""
    assert ticket_repository.get_ticket(memory_session, 999) is None


def test_create_ticket_lets_db_generate_id(memory_session):
    """A created ticket gets a db-generated id and status defaults to 'open'."""
    ticket = ticket_repository.create_ticket(
        memory_session,
        customer_name="Carol Ng",
        subject="Refund question",
        message="Please refund my order.",
    )

    assert ticket.id is not None
    assert ticket.status == "open"
    assert ticket.customer_name == "Carol Ng"

    memory_session.commit()
    assert memory_session.query(Ticket).count() == 1


def test_update_ticket_existing(memory_session):
    """Updating an existing ticket changes only the provided fields."""
    created = _create_ticket(memory_session)

    updated = ticket_repository.update_ticket(
        memory_session, created.id, status="resolved", subject="Fixed issue"
    )

    assert updated is not None
    assert updated.id == created.id
    assert updated.status == "resolved"
    assert updated.subject == "Fixed issue"
    assert updated.customer_name == "Alice Smith"  # unchanged


def test_update_ticket_missing(memory_session):
    """Updating a missing id returns None."""
    assert (
        ticket_repository.update_ticket(memory_session, 999, status="resolved")
        is None
    )


def test_delete_ticket_existing(memory_session):
    """Deleting an existing ticket returns True and removes it."""
    created = _create_ticket(memory_session)

    deleted = ticket_repository.delete_ticket(memory_session, created.id)

    assert deleted is True
    assert ticket_repository.get_ticket(memory_session, created.id) is None
    assert memory_session.query(Ticket).count() == 0


def test_delete_ticket_missing(memory_session):
    """Deleting a missing id returns False."""
    assert ticket_repository.delete_ticket(memory_session, 999) is False