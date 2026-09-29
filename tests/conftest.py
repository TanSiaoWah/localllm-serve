import pytest
from decimal import Decimal
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.data.db_models import Base, Order, Payment, Ticket
import backend.services.ticket_service as ticket_service
import backend.tools.order_tools as order_tools
import backend.tools.payment_tools as payment_tools


@pytest.fixture
def ticket():
    return {
        "subject": "Login issue",
        "message": "I cannot log into my account.",
    }


# Demo tickets seeded into the isolated test database. These mirror the data
# the existing API/CRUD tests expect to exist (e.g. GET /tickets/1).
DEMO_TICKETS = [
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
        "message": "I placed order ORD-12345 and I was charged RM1299.",
        "status": "pending",
    },
]


def _make_isolated_engine():
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
def db_storage():
    """A fresh isolated in-memory DB (session factory) seeded with demo tickets.

    The tickets table uses an INTEGER PRIMARY KEY so SQLite can auto-generate
    ids. (BIGINT is not a rowid alias in SQLite; PostgreSQL is unaffected.)
    """
    engine = _make_isolated_engine()
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

    factory = sessionmaker(bind=engine)
    # Create the order/payment tables (tickets already exists above).
    # Order/Payment reference each other, so create_all resolves the FK order.
    Base.metadata.create_all(engine, tables=[Payment.__table__, Order.__table__])

    session = factory()
    try:
        for row in DEMO_TICKETS:
            session.add(Ticket(**row))
        # Demo payment + order so DB-backed tools and the tool agent work.
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
    finally:
        session.close()

    yield factory
    engine.dispose()


@pytest.fixture(autouse=True)
def _isolate_backends(db_storage):
    """Point every DB-backed module at the isolated DB for each test.

    Without this, the ticket service and the order/payment tools (now using
    SQLAlchemy) would open sessions against the real Supabase DATABASE_URL,
    which tests must not depend on and must not mutate.
    """
    ticket_service.SessionLocal = db_storage
    order_tools.SessionLocal = db_storage
    payment_tools.SessionLocal = db_storage
    yield