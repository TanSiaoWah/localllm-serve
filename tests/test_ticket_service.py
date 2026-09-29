"""Focused unit tests for the ticket service.

These run against the isolated in-memory SQLite database provided by the
``db_storage`` / autouse conftest fixtures. They never connect to Supabase.
"""

import pytest
from fastapi import HTTPException

from backend.data.db_models import Ticket
from backend.models import TicketCreate, TicketUpdate
from backend.services import ticket_service


def _count_tickets(db_storage):
    session = db_storage()
    try:
        return session.query(Ticket).count()
    finally:
        session.close()


def test_list_tickets(db_storage):
    """list returns every seeded ticket as a dict."""
    result = ticket_service.get_all_tickets()

    assert isinstance(result, list)
    assert {t["id"] for t in result} == {1, 2, 3}
    assert result[0]["customer_name"] == "Alice Smith"


def test_get_ticket_existing(db_storage):
    """get returns the matching ticket as a dict."""
    ticket = ticket_service.get_ticket_by_id(1)

    assert ticket["id"] == 1
    assert ticket["subject"] == "Login issue"
    assert ticket["status"] == "open"


def test_get_ticket_missing(db_storage):
    """get for a missing id raises 404 (service-layer, not HTTP route)."""
    with pytest.raises(HTTPException) as exc_info:
        ticket_service.get_ticket_by_id(999)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Ticket not found"


def test_create_ticket(db_storage):
    """create persists and returns a dict; the id comes from the database."""
    created = ticket_service.create_ticket(
        TicketCreate(
            customer_name="Charlie Lee",
            subject="Shipping question",
            message="When will my order arrive?",
        )
    )

    assert created["id"] is not None
    assert created["status"] == "open"
    assert created["customer_name"] == "Charlie Lee"
    assert _count_tickets(db_storage) == 4


def test_update_ticket_existing(db_storage):
    """update changes only the provided fields."""
    updated = ticket_service.update_ticket(1, TicketUpdate(status="resolved"))

    assert updated["id"] == 1
    assert updated["status"] == "resolved"
    assert updated["customer_name"] == "Alice Smith"  # unchanged


def test_update_ticket_missing(db_storage):
    """update for a missing id raises 404."""
    with pytest.raises(HTTPException) as exc_info:
        ticket_service.update_ticket(999, TicketUpdate(status="resolved"))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Ticket not found"


def test_delete_ticket_existing(db_storage):
    """delete removes an existing ticket and reports it."""
    result = ticket_service.delete_ticket(1)

    assert result == {"message": "Ticket deleted"}
    assert _count_tickets(db_storage) == 2
    with pytest.raises(HTTPException):
        ticket_service.get_ticket_by_id(1)


def test_delete_ticket_missing(db_storage):
    """delete for a missing id raises 404."""
    with pytest.raises(HTTPException) as exc_info:
        ticket_service.delete_ticket(999)

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Ticket not found"