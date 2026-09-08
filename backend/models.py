from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# Pydantic model defining the expected fields for creating a new ticket
class TicketCreate(BaseModel):
    customer_name: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("customer_name")
    def customer_name_not_whitespace(cls, value: str) -> str:
        """Strip surrounding whitespace and reject whitespace-only names."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Customer name cannot be empty")
        return stripped

    @field_validator("subject")
    def subject_not_whitespace(cls, value: str) -> str:
        """Strip surrounding whitespace and reject whitespace-only subjects."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Subject cannot be empty")
        return stripped

    @field_validator("message")
    def message_not_whitespace(cls, value: str) -> str:
        """Strip surrounding whitespace and reject whitespace-only messages."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("Message cannot be empty")
        return stripped


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
