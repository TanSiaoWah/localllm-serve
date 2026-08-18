from fastapi import HTTPException

from backend.data.database import tickets
from backend.models import TicketCreate, TicketUpdate


def get_all_tickets():
    """Return all tickets."""
    return tickets


def get_ticket_by_id(ticket_id: int):
    """Return a single ticket by its ID, or 404 if not found."""
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            return ticket
    raise HTTPException(status_code=404, detail="Ticket not found")


def create_ticket(ticket: TicketCreate):
    """Create a new ticket and add it to the in-memory list."""
    new_id = max(t["id"] for t in tickets) + 1
    new_ticket = {
        "id": new_id,
        "customer_name": ticket.customer_name,
        "subject": ticket.subject,
        "message": ticket.message,
        "status": "open",
    }
    tickets.append(new_ticket)
    return new_ticket


def update_ticket(ticket_id: int, updated_ticket: TicketUpdate):
    """Update an existing ticket with only the provided fields."""
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            # Only update fields that were actually provided
            if updated_ticket.customer_name is not None:
                ticket["customer_name"] = updated_ticket.customer_name
            if updated_ticket.subject is not None:
                ticket["subject"] = updated_ticket.subject
            if updated_ticket.message is not None:
                ticket["message"] = updated_ticket.message
            if updated_ticket.status is not None:
                ticket["status"] = updated_ticket.status
            return ticket
    raise HTTPException(status_code=404, detail="Ticket not found")


def delete_ticket(ticket_id: int):
    """Delete a ticket by its ID, or 404 if not found."""
    for i, ticket in enumerate(tickets):
        if ticket["id"] == ticket_id:
            tickets.pop(i)
            return {"message": "Ticket deleted"}
    raise HTTPException(status_code=404, detail="Ticket not found")
