"""Evaluation dataset for the customer-support tool-calling agent.

Each case pins the mocked LLM behaviour (which tool calls the model would
emit) and the expected agent behaviour: whether a tool is called, the tool
name(s), their argument(s), and the order for multi-tool flows.

The mocked LLM keeps these tests deterministic; they never call the real
model. Tool execution runs against the isolated in-memory SQLite database
(see tests/conftest.py), so they never touch Supabase either.
"""

import json
import types

import pytest

import backend.data.order_repository as order_repository
import backend.tools.order_tools as order_tools
import backend.tools.payment_tools as payment_tools
import backend.services.tool_agent_service as tool_agent_service

# Expected verified tool results (from the seeded demo data).
ORDER_RESULT = {"product": "Laptop", "status": "shipped", "payment_id": "PAY-88888"}
PAYMENT_RESULT = {"status": "captured", "amount": 1299.0, "currency": "MYR"}


def _mk_tool_call(name, args, turn, position):
    """Build a fake tool_call namespace like the OpenAI SDK returns."""
    return types.SimpleNamespace(
        id=f"call-{turn}-{position}",
        function=types.SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _response_with_tools(tool_calls):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=None, tool_calls=tool_calls)
            )
        ]
    )


def _response_final(answer):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=answer, tool_calls=None)
            )
        ]
    )


def _build_llm(case):
    """Return a fake ``create`` that scripts the case's tool-call turns."""
    state = {"call": 0}
    turns = case["llm_turns"]

    def fake_create(**kwargs):
        index = state["call"]
        state["call"] += 1
        if index < len(turns):
            tool_calls = [
                _mk_tool_call(name, args, index, position)
                for position, (name, args) in enumerate(turns[index])
            ]
            return _response_with_tools(tool_calls)
        return _response_final(case["final_answer"])

    return fake_create


CASES = [
    {
        "scenario": "general_no_tool",
        "question": "How do I reset my password?",
        "llm_turns": [],
        "final_answer": "You can reset your password from the login page.",
        "expected": {"tools": [], "args": [], "results": []},
        "force_order_db_error": False,
    },
    {
        "scenario": "order_tool",
        "question": "Where is order ORD-12345?",
        "llm_turns": [[("get_order_status", {"order_id": "ORD-12345"})]],
        "final_answer": "Order ORD-12345 is shipped.",
        "expected": {
            "tools": ["get_order_status"],
            "args": [{"order_id": "ORD-12345"}],
            "results": [ORDER_RESULT],
        },
        "force_order_db_error": False,
    },
    {
        "scenario": "payment_tool",
        "question": "What is the status of payment PAY-88888?",
        "llm_turns": [[("get_payment_status", {"payment_id": "PAY-88888"})]],
        "final_answer": "Payment PAY-88888 is captured.",
        "expected": {
            "tools": ["get_payment_status"],
            "args": [{"payment_id": "PAY-88888"}],
            "results": [PAYMENT_RESULT],
        },
        "force_order_db_error": False,
    },
    {
        "scenario": "order_then_payment",
        "question": "What is the payment status for order ORD-12345?",
        "llm_turns": [
            [("get_order_status", {"order_id": "ORD-12345"})],
            [("get_payment_status", {"payment_id": "PAY-88888"})],
        ],
        "final_answer": "Payment PAY-88888 is captured.",
        "expected": {
            "tools": ["get_order_status", "get_payment_status"],
            "args": [{"order_id": "ORD-12345"}, {"payment_id": "PAY-88888"}],
            "results": [ORDER_RESULT, PAYMENT_RESULT],
        },
        "force_order_db_error": False,
    },
    {
        "scenario": "missing_identifier",
        "question": "What is the status of my order?",
        "llm_turns": [],
        "final_answer": "Could you please provide your order ID (it starts with ORD-)?",
        "expected": {"tools": [], "args": [], "results": []},
        "force_order_db_error": False,
    },
    {
        "scenario": "invalid_order_id",
        "question": "Check order PAY-88888",
        "llm_turns": [[("get_order_status", {"order_id": "PAY-88888"})]],
        "final_answer": "That order ID is not valid.",
        "expected": {
            "tools": ["get_order_status"],
            "args": [{"order_id": "PAY-88888"}],
            "results": [{"error": "Invalid order ID"}],
        },
        "force_order_db_error": False,
    },
    {
        "scenario": "unknown_order_id",
        "question": "Where is order ORD-99999?",
        "llm_turns": [[("get_order_status", {"order_id": "ORD-99999"})]],
        "final_answer": "Order ORD-99999 could not be found.",
        "expected": {
            "tools": ["get_order_status"],
            "args": [{"order_id": "ORD-99999"}],
            "results": [{"error": "Order not found"}],
        },
        "force_order_db_error": False,
    },
    {
        "scenario": "service_unavailable",
        "question": "Where is order ORD-12345?",
        "llm_turns": [[("get_order_status", {"order_id": "ORD-12345"})]],
        "final_answer": (
            "The order information cannot be verified right now because the "
            "order service is unavailable."
        ),
        "expected": {
            "tools": ["get_order_status"],
            "args": [{"order_id": "ORD-12345"}],
            "results": [{"error": "Order service temporarily unavailable"}],
        },
        "force_order_db_error": True,
    },
]


def _raise_order_lookup(session, order_id):
    """Simulate an unexpected database failure for the service-unavailable case."""
    raise RuntimeError("simulated database failure")


def test_eval_dataset_covers_required_scenarios():
    """The dataset covers all eight required evaluation scenarios."""
    scenarios = {case["scenario"] for case in CASES}
    assert scenarios == {
        "general_no_tool",
        "order_tool",
        "payment_tool",
        "order_then_payment",
        "missing_identifier",
        "invalid_order_id",
        "unknown_order_id",
        "service_unavailable",
    }


@pytest.mark.parametrize("case", CASES, ids=[case["scenario"] for case in CASES])
def test_agent_case(case, ticket, monkeypatch):
    """Run one evaluation case and assert the agent's tool behaviour."""
    recorded_calls = []

    real_get_order_status = order_tools.get_order_status
    real_get_payment_status = payment_tools.get_payment_status

    def spy_order(order_id):
        recorded_calls.append(
            {"tool": "get_order_status", "args": {"order_id": order_id}}
        )
        return real_get_order_status(order_id)

    def spy_payment(payment_id):
        recorded_calls.append(
            {"tool": "get_payment_status", "args": {"payment_id": payment_id}}
        )
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)
    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    if case["force_order_db_error"]:
        monkeypatch.setattr(order_repository, "get_order", _raise_order_lookup)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", _build_llm(case)
    )

    result = tool_agent_service.ask_with_tools(ticket, case["question"])

    expected = case["expected"]

    # Whether a tool was called, which tools, their arguments, and their order.
    assert [call["tool"] for call in recorded_calls] == expected["tools"]
    assert [call["args"] for call in recorded_calls] == expected["args"]

    # verified_facts reflects the executed tools, in order, with their results.
    assert [fact["tool"] for fact in result["verified_facts"]] == expected["tools"]
    assert [fact["result"] for fact in result["verified_facts"]] == expected["results"]
    assert result["answer"] == case["final_answer"]