from fastapi import APIRouter

from backend.models import TicketCreate, TicketUpdate
import backend.services.ticket_service as ticket_service

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.get("/")
def get_tickets():
    """Return the full list of tickets."""
    return ticket_service.get_all_tickets()


@router.get("/{ticket_id}")
def get_ticket(ticket_id: int):
    """Return a single ticket by its ID."""
    return ticket_service.get_ticket_by_id(ticket_id)


@router.post("/")
def create_ticket(ticket: TicketCreate):
    """Create a new ticket."""
    return ticket_service.create_ticket(ticket)


@router.put("/{ticket_id}")
def update_ticket(ticket_id: int, updated_ticket: TicketUpdate):
    """Update an existing ticket."""
    return ticket_service.update_ticket(ticket_id, updated_ticket)


@router.delete("/{ticket_id}")
def delete_ticket(ticket_id: int):
    """Delete a ticket by its ID."""
    return ticket_service.delete_ticket(ticket_id)
