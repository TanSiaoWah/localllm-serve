from fastapi import APIRouter, HTTPException

from backend.models import TicketAskResponse, TicketCreate, TicketQuestion, TicketUpdate
import backend.services.ticket_service as ticket_service
import backend.services.ai_service as ai_service
import backend.services.tool_agent_service as tool_agent_service
from backend.services.ai_service import AIServiceUnavailableError, InvalidAIResponseError
from backend.services.tool_agent_service import (
    InvalidToolCallError,
    ToolAgentMaxTurnsError,
    ToolAgentUnavailableError,
    UnknownToolError,
)

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


@router.post("/{ticket_id}/analyze")
def analyze_ticket(ticket_id: int):
    """Analyze a ticket with the local LLM."""
    ticket = ticket_service.get_ticket_by_id(ticket_id)
    try:
        analysis = ai_service.analyze_ticket(ticket)
    except InvalidAIResponseError:
        raise HTTPException(
            status_code=502,
            detail="LLM returned invalid structured output",
        )
    except AIServiceUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Local LLM service is unavailable",
        )
    return {"ticket_id": ticket_id, "analysis": analysis}


@router.post("/{ticket_id}/ask", response_model=TicketAskResponse)
def ask_ticket(ticket_id: int, request: TicketQuestion):
    """Ask the tool-capable agent a question about a ticket."""
    ticket = ticket_service.get_ticket_by_id(ticket_id)
    try:
        result = tool_agent_service.ask_with_tools(
            ticket,
            request.question,
        )
    except ToolAgentUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Local LLM service is unavailable",
        )
    except InvalidToolCallError:
        raise HTTPException(
            status_code=502,
            detail="LLM returned an invalid tool call",
        )
    except UnknownToolError:
        raise HTTPException(
            status_code=502,
            detail="LLM requested an unsupported tool",
        )
    except ToolAgentMaxTurnsError:
        raise HTTPException(
            status_code=502,
            detail="LLM tool agent exceeded maximum turns",
        )
    return {
        "ticket_id": ticket_id,
        "answer": result["answer"],
        "verified_facts": result["verified_facts"],
    }
