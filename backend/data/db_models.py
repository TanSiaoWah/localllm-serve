from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Numeric, String, Text, func
from sqlalchemy.orm import declarative_base

# Base class shared by all SQLAlchemy ORM models.
Base = declarative_base()


class Ticket(Base):
    """ORM model mapped to the existing support.tickets PostgreSQL table.

    The database schema is authoritative: Python reproduces only the columns
    and types. CHECK constraints and defaults that already live in the
    database are intentionally not recreated here.
    """

    __tablename__ = "tickets"
    __table_args__ = {"schema": "support"}

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    customer_name = Column(Text, nullable=False)
    subject = Column(Text, nullable=False)
    message = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="open", server_default="open")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=func.now(),
        server_default=func.now(),
    )


class Payment(Base):
    """ORM model mapped to the existing support.payments PostgreSQL table."""

    __tablename__ = "payments"
    __table_args__ = {"schema": "support"}

    payment_id = Column(String(64), primary_key=True)
    status = Column(Text, nullable=False)
    amount = Column(Numeric(12, 2), nullable=False)
    currency = Column(String(3), nullable=False)


class Order(Base):
    """ORM model mapped to the existing support.orders PostgreSQL table."""

    __tablename__ = "orders"
    __table_args__ = {"schema": "support"}

    order_id = Column(String(64), primary_key=True)
    product = Column(Text, nullable=False)
    status = Column(Text, nullable=False)
    payment_id = Column(
        String(64),
        ForeignKey("support.payments.payment_id"),
        nullable=False,
        unique=True,
    )