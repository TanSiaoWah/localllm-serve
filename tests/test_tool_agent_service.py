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