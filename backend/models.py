from typing import Optional

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
