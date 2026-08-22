from typing import Literal, Optional

from pydantic import BaseModel


# Pydantic model defining the expected fields for creating a new ticket
class TicketCreate(BaseModel):
    customer_name: str
    subject: str
    message: str


# Pydantic model for updating tickets — all fields are optional
class TicketUpdate(BaseModel):
    customer_name: Optional[str] = None
    subject: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None


# Pydantic model for the LLM's structured ticket analysis
class TicketAnalysis(BaseModel):
    issue: str
    category: Literal["account", "billing", "shipping", "refund", "technical", "other"]
    urgency: Literal["low", "medium", "high"]


# Pydantic model for asking the tool-capable agent a question about a ticket
class TicketQuestion(BaseModel):
    question: str


# Pydantic model for a single verified backend fact returned by a tool
class VerifiedFact(BaseModel):
    tool: str
    result: dict


# Pydantic model for the /tickets/{ticket_id}/ask response
class TicketAskResponse(BaseModel):
    ticket_id: int
    answer: str
    verified_facts: list[VerifiedFact]
