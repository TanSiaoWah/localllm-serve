"""Idempotent seed script for the Supabase ``support`` schema.

Copies the application's current demo data into the remote database:

- ``support.tickets``  <- the explicit seed-ticket fixtures below
- ``support.payments`` <- the existing payment fixture (PAY-88888)
- ``support.orders``   <- the existing order fixture (ORD-12345)

The seed is idempotent: a record is only inserted when its primary key is
not already present, so running this script multiple times never creates
duplicates. Existing rows are left untouched.

Usage (from the project root, with the virtualenv active):

    python -m backend.data.seed
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.data.database import SessionLocal
from backend.data.db_models import Order, Payment, Ticket
from backend.tools.order_tools import orders as TOOL_ORDERS
from backend.tools.payment_tools import payments as TOOL_PAYMENTS

# Demo ticket fixtures. These mirror the records previously held in the
# in-memory ticket list so the seeded database matches the application data.
SEED_TICKETS = [
    {
        "id": 1,
        "customer_name": "Alice Smith",
        "subject": "Login issue",
        "message": "I cannot log into my account.",
        "status": "open",
    },
    {
        "id": 2,
        "customer_name": "Bob Jones",
        "subject": "Billing question",
        "message": "Why was I charged twice this month?",
        "status": "pending",
    },
    {
        "id": 3,
        "customer_name": "David Tan",
        "subject": "Payment and order issue",
        "message": (
            "I placed order ORD-12345 and I was charged RM1299, but I want to "
            "confirm whether my payment was successful and what the current "
            "status of my order is."
        ),
        "status": "pending",
    },
]


def ticket_seed_rows():
    """Return the ticket fixture rows (copied so callers cannot mutate them)."""
    return [dict(ticket) for ticket in SEED_TICKETS]


def payment_seed_rows():
    """Return the payment fixture row (PAY-88888)."""
    fixture = TOOL_PAYMENTS["PAY-88888"]
    return [{"payment_id": "PAY-88888", **fixture}]


def order_seed_rows():
    """Return the order fixture row (ORD-12345)."""
    fixture = TOOL_ORDERS["ORD-12345"]
    return [{"order_id": "ORD-12345", **fixture}]


def _insert_missing(session, model, rows):
    """Insert each row unless its primary key already exists.

    Returns the number of rows actually inserted.
    """
    inserted = 0
    for row in rows:
        pk = row[model.__table__.primary_key.columns.keys()[0]]
        if session.get(model, pk) is None:
            session.add(model(**row))
            inserted += 1
    return inserted


# Advances the PostgreSQL identity sequence behind support.tickets.id so the
# next generated id is MAX(id) + 1. Explicit inserts in the seed do not advance
# the identity sequence, so without this a later insert without an id would
# collide with an existing one.
TICKET_ID_SEQUENCE_SQL = text(
    "SELECT setval("
    "pg_get_serial_sequence('support.tickets', 'id'), "
    "COALESCE((SELECT MAX(id) FROM support.tickets), 0)"
    ")"
)


def _sync_ticket_id_sequence(session) -> None:
    """Make PostgreSQL's next tickets id MAX(support.tickets.id) + 1.

    ``setval`` stores the given value and PostgreSQL then hands out that value
    plus one on the next insert, so setting it to ``MAX(id)`` produces the next
    id as ``MAX(id) + 1``. It is run after ticket seeding but is safe on every
    run (including idempotent re-runs, where it just re-sets the same value).

    Only executed on PostgreSQL; other dialects (like SQLite in tests) manage
    autoincrement differently and are left untouched.
    """
    if session.get_bind().dialect.name != "postgresql":
        return
    session.execute(TICKET_ID_SEQUENCE_SQL)


def seed(session: Session) -> int:
    """Seed all missing records and commit.

    Returns the total number of records inserted (0 if everything is
    already present, which makes repeated runs safe).
    """
    # Insert payments first so the orders foreign key can reference them.
    # The explicit flush makes sure the payment INSERT is sent to the
    # database before the order INSERT is emitted (there is no ORM
    # relationship between the models to enforce the ordering).
    inserted = 0
    inserted += _insert_missing(session, Payment, payment_seed_rows())
    session.flush()
    inserted += _insert_missing(session, Order, order_seed_rows())
    inserted += _insert_missing(session, Ticket, ticket_seed_rows())
    # Keep the tickets identity sequence in sync after the explicit id inserts.
    _sync_ticket_id_sequence(session)
    session.commit()
    return inserted


def main() -> None:
    session = SessionLocal()
    try:
        inserted = seed(session)
        print(f"Seed complete. Records inserted: {inserted}")
    finally:
        session.close()


if __name__ == "__main__":
    main()