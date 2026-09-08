import pytest
from pydantic import ValidationError

from backend.models import TicketCreate


def test_customer_name_whitespace_only_rejected():
    """A whitespace-only customer_name fails validation."""
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="   ",
            subject="Login issue",
            message="I cannot log in.",
        )


def test_subject_whitespace_only_rejected():
    """A whitespace-only subject fails validation."""
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="Alice Smith",
            subject="   ",
            message="I cannot log in.",
        )


def test_message_whitespace_only_rejected():
    """A whitespace-only message fails validation."""
    with pytest.raises(ValidationError):
        TicketCreate(
            customer_name="Alice Smith",
            subject="Login issue",
            message="   ",
        )


def test_surrounding_whitespace_stripped():
    """Surrounding whitespace is stripped from all three fields."""
    ticket = TicketCreate(
        customer_name="  Alice Smith  ",
        subject="  Login issue  ",
        message="  I cannot log in.  ",
    )
    assert ticket.customer_name == "Alice Smith"
    assert ticket.subject == "Login issue"
    assert ticket.message == "I cannot log in."