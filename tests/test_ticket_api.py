from fastapi.testclient import TestClient

from backend.main import app
import backend.services.tool_agent_service as tool_agent_service
from backend.services.tool_agent_service import (
    InvalidToolCallError,
    ToolAgentMaxTurnsError,
    ToolAgentUnavailableError,
    UnknownToolError,
)


def test_ask_ticket_with_mocked_agent(monkeypatch):
    """POST /tickets/1/ask returns the mocked agent's answer and facts."""

    def fake_ask_with_tools(ticket, question):
        return {
            "answer": "The order is shipped.",
            "verified_facts": [
                {
                    "tool": "get_order_status",
                    "result": {
                        "product": "Laptop",
                        "status": "shipped",
                        "payment_id": "PAY-88888",
                    },
                }
            ],
        }

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == 1
    assert body["answer"] == "The order is shipped."
    assert body["verified_facts"][0]["tool"] == "get_order_status"


def test_ask_ticket_agent_unavailable(monkeypatch):
    """POST /tickets/1/ask returns 503 when the LLM service is unavailable."""

    def fake_ask_with_tools(ticket, question):
        raise ToolAgentUnavailableError("Local vLLM server is unavailable")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Local LLM service is unavailable"}


def test_ask_ticket_invalid_tool_call(monkeypatch):
    """POST /tickets/1/ask returns 502 when the LLM returns an invalid tool call."""

    def fake_ask_with_tools(ticket, question):
        raise InvalidToolCallError("LLM returned an invalid tool call")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM returned an invalid tool call"}


def test_ask_ticket_unknown_tool(monkeypatch):
    """POST /tickets/1/ask returns 502 when the LLM requests an unsupported tool."""

    def fake_ask_with_tools(ticket, question):
        raise UnknownToolError("LLM requested an unknown tool")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM requested an unsupported tool"}


def test_ask_ticket_max_turns(monkeypatch):
    """POST /tickets/1/ask returns 502 when the tool agent exceeds its turn limit."""

    def fake_ask_with_tools(ticket, question):
        raise ToolAgentMaxTurnsError("Tool agent exceeded maximum turns")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM tool agent exceeded maximum turns"}
