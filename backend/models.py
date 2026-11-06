from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator


TicketSubject = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
TicketMessage = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]


# Pydantic model defining the expected fields for creating a new ticket
class TicketCreate(BaseModel):
    customer_name: str = Field(min_length=1)
    subject: TicketSubject
    message: TicketMessage

    @field_validator("customer_name")
    def customer_name_not_whitespace(cls, value: str) -> str:
        """Strip surrounding whitespace and reject whitespace-only names."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Customer name cannot be empty")
        return stripped


# Pydantic model for updating tickets — all fields are optional
class TicketUpdate(BaseModel):
    customer_name: Optional[str] = None
    subject: Optional[TicketSubject] = None
    message: Optional[TicketMessage] = None
    status: Optional[str] = None


# Pydantic model for the LLM's structured ticket analysis
class TicketAnalysis(BaseModel):
    issue: str
    category: Literal["account", "billing", "shipping", "refund", "technical", "other"]
    urgency: Literal["low", "medium", "high"]


# Pydantic model for asking the tool-capable agent a question about a ticket
class TicketQuestion(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


# Pydantic model for a single verified backend fact returned by a tool
class VerifiedFact(BaseModel):
    tool: str
    result: dict


# Pydantic model for the /tickets/{ticket_id}/ask response
class TicketAskResponse(BaseModel):
    ticket_id: int
    answer: str
    verified_facts: list[VerifiedFact]
