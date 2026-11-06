import pytest
from pydantic import ValidationError

from backend.models import TicketCreate, TicketQuestion, TicketUpdate


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


@pytest.mark.parametrize("model,field,limit", [
    (TicketCreate, "subject", 200),
    (TicketCreate, "message", 4000),
    (TicketUpdate, "subject", 200),
    (TicketUpdate, "message", 4000),
    (TicketQuestion, "question", 2000),
])
def test_input_length_boundaries_after_stripping(model, field, limit):
    data = {"customer_name": "Alice", "subject": "Issue", "message": "Help"} if model is TicketCreate else {}
    for length in (1, limit):
        result = model(**{**data, field: " \t" + "x" * length + "\n "})
        assert getattr(result, field) == "x" * length
    for value in ("", " \t\n ", "x" * (limit + 1)):
        with pytest.raises(ValidationError):
            model(**{**data, field: value})


def test_update_fields_remain_optional_and_nullable():
    assert TicketUpdate().model_dump(exclude_unset=True) == {}
    fields = {"customer_name": None, "subject": None, "message": None, "status": None}
    assert TicketUpdate(**fields).model_dump(exclude_unset=True) == fields
