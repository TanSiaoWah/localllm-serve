import logging
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError

from backend.main import app
import backend.services.tool_agent_service as tool_agent_service
from backend.services.rag_service import PolicyRetrievalError
from backend.services.tool_agent_service import (
    InvalidToolCallError,
    ToolAgentMaxTurnsError,
    ToolAgentUnavailableError,
    UnknownToolError,
)


class FakeRetriever:
    def __init__(self, results=(), error=None):
        self.results = results
        self.error = error
        self.calls = []

    def retrieve(self, question, top_k):
        self.calls.append((question, top_k))
        if self.error is not None:
            raise self.error
        return self.results


@pytest.fixture(autouse=True)
def reset_policy_retriever(monkeypatch):
    monkeypatch.setattr(app.state, "policy_retriever", None, raising=False)


@pytest.fixture(params=["unavailable", "available", "failure"])
def agent_error_retriever(request, monkeypatch):
    retriever = None
    if request.param == "available":
        retriever = FakeRetriever([({"filename": "policy.md", "index": 0, "text": "Policy."}, 0.9)])
    elif request.param == "failure":
        retriever = FakeRetriever(error=PolicyRetrievalError("simulated retrieval failure"))
    monkeypatch.setattr(app.state, "policy_retriever", retriever)
    return retriever


def test_ask_ticket_with_mocked_agent(monkeypatch):
    """POST /tickets/1/ask returns the mocked agent's answer and facts."""

    def fake_ask_with_tools(ticket, question, policy_context="", *, trace=None):
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


def test_ask_ticket_agent_unavailable(monkeypatch, agent_error_retriever):
    """POST /tickets/1/ask returns 503 when the LLM service is unavailable."""

    def fake_ask_with_tools(ticket, question, policy_context="", *, trace=None):
        raise ToolAgentUnavailableError("Local vLLM server is unavailable")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Local LLM service is unavailable"}


def test_ask_ticket_invalid_tool_call(monkeypatch, agent_error_retriever):
    """POST /tickets/1/ask returns 502 when the LLM returns an invalid tool call."""

    def fake_ask_with_tools(ticket, question, policy_context="", *, trace=None):
        raise InvalidToolCallError("LLM returned an invalid tool call")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM returned an invalid tool call"}


def test_ask_ticket_unknown_tool(monkeypatch, agent_error_retriever):
    """POST /tickets/1/ask returns 502 when the LLM requests an unsupported tool."""

    def fake_ask_with_tools(ticket, question, policy_context="", *, trace=None):
        raise UnknownToolError("LLM requested an unknown tool")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM requested an unsupported tool"}


def test_ask_ticket_max_turns(monkeypatch, agent_error_retriever):
    """POST /tickets/1/ask returns 502 when the tool agent exceeds its turn limit."""

    def fake_ask_with_tools(ticket, question, policy_context="", *, trace=None):
        raise ToolAgentMaxTurnsError("Tool agent exceeded maximum turns")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_ask_with_tools)

    client = TestClient(app)
    response = client.post(
        "/tickets/1/ask",
        json={"question": "Where is order ORD-12345?"},
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM tool agent exceeded maximum turns"}


def test_ask_ticket_forwards_ordered_policy_context_and_preserves_facts(monkeypatch):
    chunks = [
        ({"filename": "return_policy.md", "index": 3, "text": "  First passage.\n"}, 0.8),
        ({"filename": "shipping_policy.md", "index": 1, "text": "Second passage.\n- Condition."}, 0.9),
        ({"filename": "return_policy.md", "index": 0, "text": "Third passage."}, 0.7),
    ]
    retriever = FakeRetriever(chunks)
    monkeypatch.setattr(app.state, "policy_retriever", retriever)
    calls = []
    facts = [{"tool": "get_order_status", "result": {"status": "shipped"}}]

    def fake_agent(ticket, question, policy_context="", *, trace=None):
        calls.append((ticket, question, policy_context))
        return {"answer": "Supported answer.", "verified_facts": facts}

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_agent)
    question = "  Can I return my laptop after 21 days?  "
    client = TestClient(app)
    for _ in range(2):
        response = client.post("/tickets/1/ask", json={"question": question})
        assert response.status_code == 200
        assert response.json() == {
            "ticket_id": 1, "answer": "Supported answer.", "verified_facts": facts
        }
    assert retriever.calls == [(question.strip(), 3), (question.strip(), 3)]
    expected_context = (
        "[Source: return_policy.md (chunk 3)]\n  First passage.\n\n\n"
        "[Source: shipping_policy.md (chunk 1)]\nSecond passage.\n- Condition.\n\n"
        "[Source: return_policy.md (chunk 0)]\nThird passage."
    )
    assert len(calls) == 2
    for ticket, forwarded_question, context in calls:
        assert ticket["id"] == 1
        assert forwarded_question == question.strip()
        assert context == expected_context


@pytest.mark.parametrize("rag_mode", ["missing", "none", "empty", "failure"])
def test_ask_ticket_empty_context_fallback(monkeypatch, caplog, rag_mode):
    question = "Private customer question"
    private_evidence = "Private policy evidence"
    if rag_mode == "missing":
        monkeypatch.delattr(app.state, "policy_retriever")
    elif rag_mode in {"empty", "failure"}:
        error = PolicyRetrievalError(f"{question}: {private_evidence}") if rag_mode == "failure" else None
        monkeypatch.setattr(app.state, "policy_retriever", FakeRetriever(error=error))
    contexts = []

    def fake_agent(ticket, question, policy_context="", *, trace=None):
        contexts.append(policy_context)
        return {"answer": "Existing answer.", "verified_facts": []}

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_agent)
    caplog.set_level(logging.WARNING, logger="backend.api.tickets")
    response = TestClient(app).post("/tickets/1/ask", json={"question": question})
    assert response.status_code == 200
    assert response.json() == {"ticket_id": 1, "answer": "Existing answer.", "verified_facts": []}
    assert contexts == [""]
    if rag_mode == "failure":
        assert "Policy retrieval failed" in caplog.text
    assert question not in caplog.text
    assert private_evidence not in caplog.text


def test_missing_ticket_does_not_retrieve_or_call_agent(monkeypatch):
    retriever = FakeRetriever()
    monkeypatch.setattr(app.state, "policy_retriever", retriever)

    def unexpected_agent(*args, **kwargs):
        raise AssertionError("Agent called for a missing ticket")

    monkeypatch.setattr(tool_agent_service, "ask_with_tools", unexpected_agent)
    response = TestClient(app).post("/tickets/99999/ask", json={"question": "Help me."})
    assert response.status_code == 404
    assert response.json() == {"detail": "Ticket not found"}
    assert retriever.calls == []


@pytest.mark.parametrize("question", ["", " \t\n ", "x" * 2001, None])
def test_invalid_question_never_retrieves_or_calls_llm(monkeypatch, question):
    retriever = Mock()
    agent = Mock(side_effect=AssertionError("Invalid request reached agent"))
    llm = Mock(side_effect=AssertionError("Invalid request reached LLM"))
    monkeypatch.setattr(app.state, "policy_retriever", retriever)
    monkeypatch.setattr(tool_agent_service, "ask_with_tools", agent)
    monkeypatch.setattr(tool_agent_service.client.chat.completions, "create", llm)
    response = TestClient(app).post("/tickets/1/ask", json={"question": question})
    assert response.status_code == 422
    retriever.retrieve.assert_not_called()
    agent.assert_not_called()
    llm.assert_not_called()


def test_maximum_question_accepted_and_trimmed(monkeypatch):
    agent = Mock(return_value={"answer": "Answer", "verified_facts": []})
    monkeypatch.setattr(tool_agent_service, "ask_with_tools", agent)
    response = TestClient(app).post("/tickets/1/ask", json={"question": "  " + "x" * 2000 + "  "})
    assert response.status_code == 200
    assert agent.call_args.args[1] == "x" * 2000


def test_ask_timeout_returns_service_unavailable(monkeypatch):
    llm = Mock(side_effect=APITimeoutError(request=None))
    monkeypatch.setattr(tool_agent_service.client.chat.completions, "create", llm)
    response = TestClient(app).post("/tickets/1/ask", json={"question": "Help me"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Local LLM service is unavailable"}
