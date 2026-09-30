import logging
import types

import pytest
from openai import APIConnectionError

import backend.services.tool_agent_service as tool_agent_service
from backend.services.tool_agent_service import (
    InvalidToolCallError,
    ToolAgentMaxTurnsError,
    ToolAgentUnavailableError,
    UnknownToolError,
)


def test_ask_with_tools_no_tool_requested(monkeypatch, ticket):
    """ask_with_tools() returns the plain answer when the LLM does not request a tool."""

    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="Hello!",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return fake_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Say hello in one sentence.",
    )

    assert result["answer"] == "Hello!"
    assert result["verified_facts"] == []


def test_ask_with_tools_single_tool_call(monkeypatch, ticket):
    """ask_with_tools() executes a single order-status tool call and returns the final answer."""

    call_count = [0]

    fake_first_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-1",
                            function=types.SimpleNamespace(
                                name="get_order_status",
                                arguments='{"order_id": "ORD-12345"}',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    fake_second_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="The order is shipped.",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return fake_first_response
        return fake_second_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Where is order ORD-12345?",
    )

    assert result["answer"] == "The order is shipped."
    assert len(result["verified_facts"]) == 1
    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][0]["result"]["status"] == "shipped"


def test_ask_with_tools_chained_tool_calls(monkeypatch, ticket):
    """ask_with_tools() chains get_order_status and get_payment_status across turns."""

    first_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-order",
                            function=types.SimpleNamespace(
                                name="get_order_status",
                                arguments='{"order_id": "ORD-12345"}',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    second_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-payment",
                            function=types.SimpleNamespace(
                                name="get_payment_status",
                                arguments='{"payment_id": "PAY-88888"}',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    third_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="The payment was captured.",
                    tool_calls=None,
                )
            )
        ]
    )

    responses = [first_response, second_response, third_response]

    def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Was the payment for order ORD-12345 successful?",
    )

    assert result["answer"] == "The payment was captured."
    assert len(result["verified_facts"]) == 2
    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][1]["tool"] == "get_payment_status"
    assert result["verified_facts"][1]["result"]["status"] == "captured"


def test_ask_with_tools_parallel_tool_calls(monkeypatch, ticket):
    """ask_with_tools() executes two tool calls requested in the same turn."""

    first_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-order",
                            function=types.SimpleNamespace(
                                name="get_order_status",
                                arguments='{"order_id": "ORD-12345"}',
                            ),
                        ),
                        types.SimpleNamespace(
                            id="call-payment",
                            function=types.SimpleNamespace(
                                name="get_payment_status",
                                arguments='{"payment_id": "PAY-88888"}',
                            ),
                        ),
                    ],
                )
            )
        ]
    )

    second_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="The order is shipped and the payment was captured.",
                    tool_calls=None,
                )
            )
        ]
    )

    responses = [first_response, second_response]

    def fake_create(**kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Check order ORD-12345 and payment PAY-88888.",
    )

    assert result["answer"] == "The order is shipped and the payment was captured."
    assert len(result["verified_facts"]) == 2
    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][1]["tool"] == "get_payment_status"


def test_ask_with_tools_malformed_tool_arguments(monkeypatch, ticket):
    """ask_with_tools() raises InvalidToolCallError when tool arguments are malformed JSON."""

    first_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-invalid",
                            function=types.SimpleNamespace(
                                name="get_order_status",
                                arguments='{"order_id": "ORD-12345"',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return first_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    with pytest.raises(InvalidToolCallError):
        tool_agent_service.ask_with_tools(
            ticket,
            "Where is order ORD-12345?",
        )


def test_ask_with_tools_unknown_tool(monkeypatch, ticket):
    """ask_with_tools() raises UnknownToolError when the LLM requests an unsupported tool."""

    first_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-unknown",
                            function=types.SimpleNamespace(
                                name="delete_customer_account",
                                arguments="{}",
                            ),
                        )
                    ],
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return first_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    with pytest.raises(UnknownToolError):
        tool_agent_service.ask_with_tools(
            ticket,
            "Delete the customer account.",
        )


def test_ask_with_tools_missing_required_argument(monkeypatch, ticket):
    """ask_with_tools() raises InvalidToolCallError when a required argument is missing."""

    first_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-missing-arg",
                            function=types.SimpleNamespace(
                                name="get_order_status",
                                arguments="{}",
                            ),
                        )
                    ],
                )
            )
        ]
    )

    def fake_create(**kwargs):
        return first_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    with pytest.raises(InvalidToolCallError):
        tool_agent_service.ask_with_tools(
            ticket,
            "Where is my order?",
        )


def test_ask_with_tools_connection_error(monkeypatch, ticket):
    """ask_with_tools() raises ToolAgentUnavailableError when the LLM server is unreachable."""

    def fake_create(**kwargs):
        raise APIConnectionError(request=None)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    with pytest.raises(ToolAgentUnavailableError):
        tool_agent_service.ask_with_tools(
            ticket,
            "Where is order ORD-12345?",
        )


def test_ask_with_tools_max_turns(monkeypatch, ticket):
    """ask_with_tools() raises ToolAgentMaxTurnsError when the LLM keeps requesting tools."""

    repeat_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-repeat",
                            function=types.SimpleNamespace(
                                name="get_order_status",
                                arguments='{"order_id": "ORD-12345"}',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    call_count = [0]

    def fake_create(**kwargs):
        call_count[0] += 1
        return repeat_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    with pytest.raises(ToolAgentMaxTurnsError):
        tool_agent_service.ask_with_tools(
            ticket,
            "Keep checking order ORD-12345.",
        )

    assert call_count[0] == 5


def _tool_call_response(name, arguments):
    """Build a fake LLM response that requests a single tool call."""
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            id="call-1",
                            function=types.SimpleNamespace(
                                name=name,
                                arguments=arguments,
                            ),
                        )
                    ],
                )
            )
        ]
    )


def _final_response(answer):
    """Build a fake LLM response with a plain final answer."""
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=answer, tool_calls=None)
            )
        ]
    )


def test_ask_with_tools_payment_service_unavailable(monkeypatch, ticket):
    """A payment "service unavailable" error is reported to the model, not invented."""

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return _tool_call_response(
                "get_payment_status", '{"payment_id": "PAY-88888"}'
            )
        return _final_response(
            "The payment could not be verified because the payment service is "
            "temporarily unavailable."
        )

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )
    monkeypatch.setattr(
        tool_agent_service,
        "get_payment_status",
        lambda payment_id: {"error": "Payment service temporarily unavailable"},
    )

    result = tool_agent_service.ask_with_tools(ticket, "What is my payment status?")

    assert result["verified_facts"][0]["tool"] == "get_payment_status"
    assert result["verified_facts"][0]["result"] == {
        "error": "Payment service temporarily unavailable"
    }
    # The error is forwarded to the model so it can answer without inventing data.
    assert "temporarily unavailable" in calls[1][-1]["content"]


def test_ask_with_tools_order_service_unavailable(monkeypatch, ticket):
    """An order "service unavailable" error is reported to the model, not invented."""

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return _tool_call_response(
                "get_order_status", '{"order_id": "ORD-12345"}'
            )
        return _final_response(
            "The order could not be verified because the order service is "
            "temporarily unavailable."
        )

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )
    monkeypatch.setattr(
        tool_agent_service,
        "get_order_status",
        lambda order_id: {"error": "Order service temporarily unavailable"},
    )

    result = tool_agent_service.ask_with_tools(ticket, "Where is order ORD-12345?")

    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][0]["result"] == {
        "error": "Order service temporarily unavailable"
    }
    assert "temporarily unavailable" in calls[1][-1]["content"]


def test_ask_with_tools_tool_not_found(monkeypatch, ticket):
    """A tool "not found" error is reported to the model, not invented."""

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return _tool_call_response(
                "get_order_status", '{"order_id": "ORD-99999"}'
            )
        return _final_response("Order ORD-99999 could not be found.")

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    # Uses the real tool against the isolated test DB, which has no ORD-99999.
    result = tool_agent_service.ask_with_tools(ticket, "Where is order ORD-99999?")

    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][0]["result"] == {"error": "Order not found"}
    assert "Order not found" in calls[1][-1]["content"]


def test_ask_with_tools_logs_tool_request_and_success(monkeypatch, ticket, caplog):
    """A requested tool name and its successful outcome are logged."""
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return _tool_call_response(
                "get_order_status", '{"order_id": "ORD-12345"}'
            )
        return _final_response("The order is shipped.")

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )
    caplog.set_level(logging.INFO, logger="backend.services.tool_agent_service")

    tool_agent_service.ask_with_tools(ticket, "Where is order ORD-12345?")

    assert "Tool call requested by model: get_order_status" in caplog.text
    assert "Tool get_order_status completed successfully" in caplog.text
    # The full result/record must not be logged.
    assert "Laptop" not in caplog.text


def test_ask_with_tools_logs_tool_error(monkeypatch, ticket, caplog):
    """A tool that returns an error is logged as an error outcome."""
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        if len(calls) == 1:
            return _tool_call_response(
                "get_order_status", '{"order_id": "ORD-12345"}'
            )
        return _final_response("The order could not be verified.")

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )
    monkeypatch.setattr(
        tool_agent_service,
        "get_order_status",
        lambda order_id: {"error": "Order service temporarily unavailable"},
    )
    caplog.set_level(logging.INFO, logger="backend.services.tool_agent_service")

    tool_agent_service.ask_with_tools(ticket, "Where is order ORD-12345?")

    assert "Tool get_order_status returned an error" in caplog.text


def test_ask_with_tools_logs_max_turns(monkeypatch, ticket, caplog):
    """Reaching the maximum number of turns is logged."""
    repeat_response = _tool_call_response(
        "get_order_status", '{"order_id": "ORD-12345"}'
    )

    def fake_create(**kwargs):
        return repeat_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )
    caplog.set_level(logging.INFO, logger="backend.services.tool_agent_service")

    with pytest.raises(ToolAgentMaxTurnsError):
        tool_agent_service.ask_with_tools(ticket, "Keep checking order ORD-12345.")

    assert "Tool agent reached maximum turns" in caplog.text


def test_identifier_validation_matching_order_id_executes(monkeypatch, ticket):
    """A model-requested order ID equal to the customer's ID is executed."""
    executed = []
    real_get_order_status = tool_agent_service.get_order_status

    def spy_order(order_id):
        executed.append(order_id)
        return real_get_order_status(order_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)

    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _final_response("The order is shipped."),
    ]
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Where is order ORD-12345?",
    )

    assert executed == ["ORD-12345"]
    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][0]["result"]["status"] == "shipped"


def test_identifier_validation_mismatched_order_id_rejected(monkeypatch, ticket):
    """A model-requested order ID different from the customer's is rejected."""
    executed = []
    real_get_order_status = tool_agent_service.get_order_status

    def spy_order(order_id):
        executed.append(order_id)
        return real_get_order_status(order_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)

    calls = []
    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _final_response("I could not check that order."),
    ]

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        return responses.pop(0)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Please check ORD-99999.",
    )

    # The order tool was NOT executed and no order data was returned.
    assert executed == []
    assert result["verified_facts"][0]["tool"] == "get_order_status"
    assert result["verified_facts"][0]["result"] == {
        "error": (
            "The requested order ID does not match the order ID "
            "provided by the customer."
        )
    }
    # The controlled error goes back to the model so the loop can continue.
    assert "does not match" in calls[1][-1]["content"]
    assert "Laptop" not in calls[1][-1]["content"]


def test_identifier_validation_matching_payment_id_executes(monkeypatch, ticket):
    """A model-requested payment ID equal to the customer's ID is executed."""
    executed = []
    real_get_payment_status = tool_agent_service.get_payment_status

    def spy_payment(payment_id):
        executed.append(payment_id)
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    responses = [
        _tool_call_response(
            "get_payment_status", '{"payment_id": "PAY-88888"}'
        ),
        _final_response("The payment was captured."),
    ]
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "What is the status of payment PAY-88888?",
    )

    assert executed == ["PAY-88888"]
    assert result["verified_facts"][0]["tool"] == "get_payment_status"
    assert result["verified_facts"][0]["result"]["status"] == "captured"


def test_identifier_validation_mismatched_payment_id_rejected(monkeypatch, ticket):
    """A model-requested payment ID different from the customer's is rejected."""
    executed = []
    real_get_payment_status = tool_agent_service.get_payment_status

    def spy_payment(payment_id):
        executed.append(payment_id)
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    calls = []
    responses = [
        _tool_call_response(
            "get_payment_status", '{"payment_id": "PAY-88888"}'
        ),
        _final_response("I could not check that payment."),
    ]

    def fake_create(**kwargs):
        calls.append(kwargs["messages"])
        return responses.pop(0)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Please check payment PAY-99999.",
    )

    # The payment tool was NOT executed and no payment data was returned.
    assert executed == []
    assert result["verified_facts"][0]["tool"] == "get_payment_status"
    assert result["verified_facts"][0]["result"] == {
        "error": (
            "The requested payment ID does not match the payment ID "
            "provided by the customer."
        )
    }
    # The controlled error goes back to the model so the loop can continue.
    assert "does not match" in calls[1][-1]["content"]
    assert "captured" not in calls[1][-1]["content"]


def test_identifier_validation_skipped_without_customer_id(monkeypatch, ticket):
    """With no ORD-/PAY- identifier in the question, validation is not applied."""
    executed = []
    real_get_order_status = tool_agent_service.get_order_status

    def spy_order(order_id):
        executed.append(order_id)
        return real_get_order_status(order_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)

    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _final_response("The order is shipped."),
    ]
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Where is my order?",
    )

    assert executed == ["ORD-12345"]
    assert result["verified_facts"][0]["result"]["status"] == "shipped"


def test_identifier_validation_preserves_order_then_payment_flow(
    monkeypatch, ticket
):
    """The order→payment flow still runs with the customer's order ID."""
    executed_orders = []
    executed_payments = []
    real_get_order_status = tool_agent_service.get_order_status
    real_get_payment_status = tool_agent_service.get_payment_status

    def spy_order(order_id):
        executed_orders.append(order_id)
        return real_get_order_status(order_id)

    def spy_payment(payment_id):
        executed_payments.append(payment_id)
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)
    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _tool_call_response(
            "get_payment_status", '{"payment_id": "PAY-88888"}'
        ),
        _final_response("The payment was captured."),
    ]
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Was the payment for order ORD-12345 successful?",
    )

    # get_order_status(ORD-12345) ran first, then
    # get_payment_status(PAY-88888) using the payment_id it returned.
    assert executed_orders == ["ORD-12345"]
    assert executed_payments == ["PAY-88888"]
    assert [fact["tool"] for fact in result["verified_facts"]] == [
        "get_order_status",
        "get_payment_status",
    ]
    assert result["verified_facts"][1]["result"]["status"] == "captured"


def test_payment_scope_order_status_only_exposes_only_order_tool(
    monkeypatch, ticket
):
    """An order-status-only question exposes only get_order_status; payment tool is excluded."""
    executed_orders = []
    executed_payments = []
    recorded_tools = []
    real_get_order_status = tool_agent_service.get_order_status
    real_get_payment_status = tool_agent_service.get_payment_status

    def spy_order(order_id):
        executed_orders.append(order_id)
        return real_get_order_status(order_id)

    def spy_payment(payment_id):
        executed_payments.append(payment_id)
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)
    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _final_response("Order ORD-12345 is shipped."),
    ]

    def fake_create(**kwargs):
        recorded_tools.append(kwargs.get("tools"))
        return responses.pop(0)

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        fake_create,
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "What is the current status of ORD-12345?",
    )

    # get_order_status was the only tool exposed across both turns.
    assert len(recorded_tools) == 2
    for tools_arg in recorded_tools:
        assert tools_arg is not None
        assert [t["function"]["name"] for t in tools_arg] == ["get_order_status"]

    assert executed_orders == ["ORD-12345"]
    assert executed_payments == []
    assert [fact["tool"] for fact in result["verified_facts"]] == [
        "get_order_status",
    ]
    assert result["verified_facts"][0]["result"]["status"] == "shipped"


def test_payment_scope_payment_question_with_order_id_allowed(
    monkeypatch, ticket
):
    """A payment question with an order ID still runs order -> payment."""
    executed_orders = []
    executed_payments = []
    real_get_order_status = tool_agent_service.get_order_status
    real_get_payment_status = tool_agent_service.get_payment_status

    def spy_order(order_id):
        executed_orders.append(order_id)
        return real_get_order_status(order_id)

    def spy_payment(payment_id):
        executed_payments.append(payment_id)
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)
    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _tool_call_response(
            "get_payment_status", '{"payment_id": "PAY-88888"}'
        ),
        _final_response("The payment was captured."),
    ]
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Was the payment for ORD-12345 successful?",
    )

    assert executed_orders == ["ORD-12345"]
    assert executed_payments == ["PAY-88888"]
    assert [fact["tool"] for fact in result["verified_facts"]] == [
        "get_order_status",
        "get_payment_status",
    ]
    assert result["verified_facts"][1]["result"]["status"] == "captured"


def test_payment_scope_duplicate_charge_question_allows_payment_tool(
    monkeypatch, ticket
):
    """A "charged" question about an order still allows the payment lookup."""
    executed_payments = []
    real_get_payment_status = tool_agent_service.get_payment_status

    def spy_payment(payment_id):
        executed_payments.append(payment_id)
        return real_get_payment_status(payment_id)

    monkeypatch.setattr(tool_agent_service, "get_payment_status", spy_payment)

    responses = [
        _tool_call_response("get_order_status", '{"order_id": "ORD-12345"}'),
        _tool_call_response(
            "get_payment_status", '{"payment_id": "PAY-88888"}'
        ),
        _final_response("The payment was captured once for RM1299.00."),
    ]
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: responses.pop(0),
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Why was I charged for ORD-12345?",
    )

    # "charged" is payment intent, so the payment tool executed.
    assert executed_payments == ["PAY-88888"]
    assert result["verified_facts"][1]["tool"] == "get_payment_status"
    assert result["verified_facts"][1]["result"]["status"] == "captured"


def test_repeated_mismatched_order_id_never_executes(monkeypatch, ticket, caplog):
    """A repeatedly mismatched order ID is always rejected, never executed."""
    executed = []
    real_get_order_status = tool_agent_service.get_order_status

    def spy_order(order_id):
        executed.append(order_id)
        return real_get_order_status(order_id)

    monkeypatch.setattr(tool_agent_service, "get_order_status", spy_order)

    # The model keeps requesting ORD-12345 on every turn.
    mismatched = _tool_call_response(
        "get_order_status", '{"order_id": "ORD-12345"}'
    )
    monkeypatch.setattr(
        tool_agent_service.client.chat.completions,
        "create",
        lambda **kwargs: mismatched,
    )
    caplog.set_level(logging.WARNING, logger="backend.services.tool_agent_service")

    # The loop terminates naturally at MAX_TURNS since the model never corrects.
    with pytest.raises(ToolAgentMaxTurnsError):
        tool_agent_service.ask_with_tools(
            ticket,
            "Please check ORD-99999.",
        )

    # The real order tool never executed the wrong identifier.
    assert executed == []
    # First rejection and the repeat rejection are both logged.
    assert "Tool call rejected: requested order ID does not" in caplog.text
    assert "Tool call rejected again: repeated mismatched order identifier" in caplog.text


def test_dynamic_tools_order_only_exposes_order_tool(monkeypatch, ticket):
    """An order-only question exposes only get_order_status to the model."""
    recorded_tools = []
    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="Order ORD-12345 is shipped.",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        recorded_tools.append(kwargs.get("tools"))
        return fake_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "What is the current status of my order ORD-12345?",
    )

    assert len(recorded_tools) == 1
    assert recorded_tools[0] is not None
    assert [t["function"]["name"] for t in recorded_tools[0]] == ["get_order_status"]
    assert result["answer"] == "Order ORD-12345 is shipped."


def test_dynamic_tools_payment_only_exposes_payment_tool(monkeypatch, ticket):
    """A question with a payment ID exposes only get_payment_status to the model."""
    recorded_tools = []
    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="Payment PAY-88888 is captured.",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        recorded_tools.append(kwargs.get("tools"))
        return fake_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "What is the status of payment PAY-88888?",
    )

    assert len(recorded_tools) == 1
    assert recorded_tools[0] is not None
    assert [t["function"]["name"] for t in recorded_tools[0]] == ["get_payment_status"]
    assert result["answer"] == "Payment PAY-88888 is captured."


def test_dynamic_tools_payment_for_order_exposes_both_tools(monkeypatch, ticket):
    """A payment question with an order ID exposes both tools to the model."""
    recorded_tools = []
    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="The order is shipped and payment was captured.",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        recorded_tools.append(kwargs.get("tools"))
        return fake_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Was the payment for ORD-12345 successful?",
    )

    assert len(recorded_tools) == 1
    assert recorded_tools[0] is not None
    assert [t["function"]["name"] for t in recorded_tools[0]] == [
        "get_order_status",
        "get_payment_status",
    ]


def test_dynamic_tools_missing_identifier_payment_question_exposes_no_tools(
    monkeypatch, ticket
):
    """A payment question without order/payment ID exposes no business tools to Qwen."""
    recorded_kwargs = []
    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="Could you provide your order or payment ID?",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        recorded_kwargs.append(kwargs)
        return fake_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "Why was I charged twice this month?",
    )

    assert len(recorded_kwargs) == 1
    assert recorded_kwargs[0].get("tools") is None
    assert recorded_kwargs[0].get("tool_choice") is None
    assert result["verified_facts"] == []
    assert result["answer"] == "Could you provide your order or payment ID?"


def test_dynamic_tools_general_login_question_exposes_no_tools(monkeypatch, ticket):
    """A general question with no identifiers exposes no business tools to Qwen."""
    recorded_kwargs = []
    fake_response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content="You can try resetting your password.",
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create(**kwargs):
        recorded_kwargs.append(kwargs)
        return fake_response

    monkeypatch.setattr(
        tool_agent_service.client.chat.completions, "create", fake_create
    )

    result = tool_agent_service.ask_with_tools(
        ticket,
        "I cannot log into my account. What can I try?",
    )

    assert len(recorded_kwargs) == 1
    assert recorded_kwargs[0].get("tools") is None
    assert recorded_kwargs[0].get("tool_choice") is None
    assert result["verified_facts"] == []
    assert result["answer"] == "You can try resetting your password."