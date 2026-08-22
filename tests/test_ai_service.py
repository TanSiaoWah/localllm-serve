import types

import pytest
from openai import APIConnectionError

import backend.services.ai_service as ai_service
from backend.services.ai_service import (
    AIServiceUnavailableError,
    InvalidAIResponseError,
)


def test_analyze_ticket_valid_response(monkeypatch, ticket):
    """analyze_ticket() parses the LLM's structured JSON into a TicketAnalysis."""

    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content='{"issue":"Cannot log into account","category":"account","urgency":"high"}'
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return fake_response

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)

    result = ai_service.analyze_ticket(ticket)

    assert result.issue == "Cannot log into account"
    assert result.category == "account"
    assert result.urgency == "high"


def test_analyze_ticket_invalid_json(monkeypatch, ticket):
    """analyze_ticket() raises InvalidAIResponseError when the LLM returns non-JSON."""

    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="this is not json",
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return fake_response

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)

    with pytest.raises(InvalidAIResponseError):
        ai_service.analyze_ticket(ticket)


def test_analyze_ticket_connection_error(monkeypatch, ticket):
    """analyze_ticket() raises AIServiceUnavailableError when the LLM server is unreachable."""

    def fake_create(**kwargs):
        raise APIConnectionError(request=None)

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)

    with pytest.raises(AIServiceUnavailableError):
        ai_service.analyze_ticket(ticket)


def test_analyze_ticket_schema_validation_failure(monkeypatch, ticket):
    """analyze_ticket() raises InvalidAIResponseError when the JSON fails schema validation."""

    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content='{"issue": "Login issue", "category": "authentication", "urgency": "critical"}',
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return fake_response

    monkeypatch.setattr(ai_service.client.chat.completions, "create", fake_create)

    with pytest.raises(InvalidAIResponseError):
        ai_service.analyze_ticket(ticket)