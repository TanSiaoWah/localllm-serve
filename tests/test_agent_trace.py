"""Offline trace tests: no inference server, MCP transport, or production DB."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import httpx
import pytest
from openai import APIConnectionError

from backend.services import agent_trace, tool_agent_service as agent
from backend.tools.kapruka_tools import KaprukaMCPError, KaprukaValidationError


def response(*calls, answer="Safe answer"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=answer, tool_calls=list(calls) or None,
    ))])


def tool(name="get_order_status", arguments='{"order_id": "ORD-12345"}'):
    return SimpleNamespace(
        id="private-call-id",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.fixture(autouse=True)
def external_guards(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=agent_trace.logger.name)
    llm = Mock(side_effect=AssertionError("Unexpected LLM call"))
    search = Mock(side_effect=AssertionError("Unexpected MCP call"))
    monkeypatch.setattr(agent.client.chat.completions, "create", llm)
    monkeypatch.setattr(agent, "search_kapruka_products", search)
    return llm, search


def events(caplog):
    records = [r for r in caplog.records if r.name == agent_trace.logger.name]
    assert all(r.exc_info is None and r.stack_info is None for r in records)
    return [json.loads(r.getMessage()) for r in records]


def assert_trace(logs, operations, status="success", category=None):
    assert logs[0]["event"] == "agent_trace_started"
    assert logs[-1]["event"] == "agent_trace_completed"
    assert len({e["trace_id"] for e in logs}) == 1
    UUID(logs[0]["trace_id"])
    steps = logs[1:-1]
    assert all(e["event"] == "agent_step_completed" for e in steps)
    assert [e["step_index"] for e in steps] == list(range(1, len(steps) + 1))
    assert [e["operation"] for e in steps] == operations
    assert all(e["duration_ms"] >= 0 for e in steps)
    assert logs[-1]["total_duration_ms"] >= sum(e["duration_ms"] for e in steps)
    assert logs[-1]["step_count"] == len(steps)
    assert logs[-1]["status"] == status
    assert logs[-1]["error_category"] == category
    return steps


LLM = "chat.completions.create"


def test_plain_answer_and_unique_executions(ticket, external_guards, caplog):
    llm, _ = external_guards
    llm.side_effect = [response(), response()]
    for _ in range(2):
        assert agent.ask_with_tools(ticket, "Hello") == {
            "answer": "Safe answer", "verified_facts": [],
        }
    logs = events(caplog)
    first = assert_trace(logs[:3], [LLM])
    second = assert_trace(logs[3:], [LLM])
    assert first[0]["step_type"] == second[0]["step_type"] == "llm"
    assert first[0]["status"] == second[0]["status"] == "success"
    assert logs[0]["trace_id"] != logs[3]["trace_id"]


def test_order_payment_sequence_and_response_unchanged(ticket, external_guards, caplog):
    llm, _ = external_guards
    llm.side_effect = [
        response(tool()),
        response(tool("get_payment_status", '{"payment_id": "PAY-88888"}')),
        response(),
    ]
    result = agent.ask_with_tools(ticket, "Check payment for ORD-12345", "Private policy")
    assert result == {
        "answer": "Safe answer",
        "verified_facts": [
            {"tool": "get_order_status", "result": {
                "product": "Laptop", "status": "shipped", "payment_id": "PAY-88888",
            }},
            {"tool": "get_payment_status", "result": {
                "status": "captured", "amount": 1299.0, "currency": "MYR",
            }},
        ],
    }
    steps = assert_trace(events(caplog), [LLM, "get_order_status", LLM, "get_payment_status", LLM])
    assert [s["step_type"] for s in steps] == ["llm", "tool", "llm", "tool", "llm"]
    assert all(s["status"] == "success" and s["error_category"] is None for s in steps)


@pytest.mark.parametrize("connected", [False, True])
def test_llm_failure_propagates_and_finalizes(ticket, external_guards, caplog, connected):
    llm, _ = external_guards
    error = (RuntimeError("private model response") if connected else
             APIConnectionError(request=httpx.Request("POST", "http://localhost/v1")))
    llm.side_effect = error
    expected = RuntimeError if connected else agent.ToolAgentUnavailableError
    with pytest.raises(expected) as raised:
        agent.ask_with_tools(ticket, "Hello")
    if connected:
        assert raised.value is error
    else:
        assert raised.value.__cause__ is error
    steps = assert_trace(events(caplog), [LLM], "error",
                         "unexpected_error" if connected else "llm_unavailable")
    assert steps[0]["status"] == "error"
    assert steps[0]["error_category"] == "llm_error"
    assert "private model response" not in caplog.text


def test_tool_exception_propagates_and_finalizes(monkeypatch, ticket, external_guards, caplog):
    llm, _ = external_guards
    llm.side_effect = [response(tool())]
    error = RuntimeError("private database credentials")
    monkeypatch.setattr(agent, "get_order_status", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as raised:
        agent.ask_with_tools(ticket, "Where is ORD-12345?")
    assert raised.value is error
    steps = assert_trace(events(caplog), [LLM, "get_order_status"], "error", "unexpected_error")
    assert steps[-1]["error_category"] == "tool_error"
    assert steps[-1]["status"] == "error"
    assert "private database credentials" not in caplog.text


def test_failure_on_later_llm_turn_then_fresh_success(ticket, external_guards, caplog):
    llm, _ = external_guards
    error = RuntimeError("private later failure")
    llm.side_effect = [response(tool()), error, response()]
    with pytest.raises(RuntimeError) as raised:
        agent.ask_with_tools(ticket, "Where is ORD-12345?")
    assert raised.value is error
    failed = events(caplog)
    steps = assert_trace(failed, [LLM, "get_order_status", LLM], "error", "unexpected_error")
    assert [s["status"] for s in steps] == ["success", "success", "error"]
    caplog.clear()
    assert agent.ask_with_tools(ticket, "Hello") == {"answer": "Safe answer", "verified_facts": []}
    successful = events(caplog)
    assert_trace(successful, [LLM])
    assert failed[0]["trace_id"] != successful[0]["trace_id"]


def test_multiple_tools_in_one_turn_preserve_execution_order(ticket, external_guards, caplog):
    llm, _ = external_guards
    llm.side_effect = [response(
        tool(), tool("get_payment_status", '{"payment_id": "PAY-88888"}'),
    ), response()]
    result = agent.ask_with_tools(ticket, "Check payment for ORD-12345")
    assert [fact["tool"] for fact in result["verified_facts"]] == [
        "get_order_status", "get_payment_status",
    ]
    assert_trace(events(caplog), [LLM, "get_order_status", "get_payment_status", LLM])


@pytest.mark.parametrize("name,arguments,question", [
    ("get_order_status", '{"order_id": "ORD-99999"}', "Where is ORD-99999?"),
    ("get_payment_status", '{"payment_id": "PAY-99999"}', "Check payment PAY-99999"),
])
def test_tool_error_result_is_failed_step(ticket, external_guards, caplog, name, arguments, question):
    llm, _ = external_guards
    llm.side_effect = [response(tool(name, arguments)), response()]
    result = agent.ask_with_tools(ticket, question)
    assert "error" in result["verified_facts"][0]["result"]
    steps = assert_trace(events(caplog), [LLM, name, LLM])
    assert steps[1]["status"] == "error"
    assert steps[1]["error_category"] == "tool_result_error"


@pytest.mark.parametrize("failure,category", [
    (None, None), (KaprukaValidationError, "validation_error"),
    (KaprukaMCPError, "service_unavailable"),
])
def test_kapruka_tracing_preserves_dispatch_and_results(
    ticket, external_guards, caplog, failure, category,
):
    llm, search = external_guards
    llm.side_effect = [
        response(tool("kapruka_search_products", '{"query": "private product query"}')),
        response(),
    ]
    search.side_effect = failure("private MCP response") if failure else None
    search.return_value = {"results": [{"name": "Private product", "description": "Private description"}]}
    result = agent.ask_with_tools(ticket, "Search Kapruka for laptops.")
    search.assert_called_once_with("private product query", 3, "LKR")
    fact = result["verified_facts"][0]
    assert fact["tool"] == "kapruka_search_products"
    assert fact["result"]["source"] == "kapruka"
    assert fact["result"]["data_type"] == "product_catalog"
    if failure:
        assert fact["result"]["error"] == (
            "Invalid Kapruka product search arguments." if failure is KaprukaValidationError
            else "Kapruka product service temporarily unavailable."
        )
    else:
        assert fact["result"]["results"] == search.return_value["results"]
    steps = assert_trace(events(caplog), [LLM, "kapruka_search_products", LLM])
    assert steps[1]["error_category"] == category
    assert steps[1]["status"] == ("error" if failure else "success")
    assert "private" not in caplog.text.lower()


@pytest.mark.parametrize("name", [
    "get_payment_status", "kapruka_search_products", "kapruka_create_order",
    "kapruka_track_order", "private attacker-controlled tool name",
])
def test_whole_batch_authorized_before_execution(
    monkeypatch, ticket, external_guards, caplog, name,
):
    llm, search = external_guards
    # Even the authorized first tool must not execute if the batch is rejected.
    llm.side_effect = [response(tool(), tool(name, "private malformed arguments"))]
    order, payment = Mock(), Mock()
    monkeypatch.setattr(agent, "get_order_status", order)
    monkeypatch.setattr(agent, "get_payment_status", payment)
    with pytest.raises(agent.UnknownToolError):
        agent.ask_with_tools(ticket, "Where is ORD-12345?")
    order.assert_not_called()
    payment.assert_not_called()
    search.assert_not_called()
    assert_trace(events(caplog), [LLM], "error", "unauthorized_tool")
    assert "private" not in caplog.text


@pytest.mark.parametrize("arguments", ["not JSON", "[]", "{}"])
def test_invalid_arguments_finalize_without_executing(
    monkeypatch, ticket, external_guards, caplog, arguments,
):
    llm, _ = external_guards
    llm.side_effect = [response(tool(arguments=arguments))]
    order = Mock()
    monkeypatch.setattr(agent, "get_order_status", order)
    with pytest.raises(agent.InvalidToolCallError):
        agent.ask_with_tools(ticket, "Where is ORD-12345?")
    order.assert_not_called()
    assert_trace(events(caplog), [LLM], "error", "invalid_tool_call")


def test_max_turns_and_mismatch_do_not_execute_tool(
    monkeypatch, ticket, external_guards, caplog,
):
    llm, _ = external_guards
    llm.side_effect = [response(tool(arguments='{"order_id": "ORD-99999"}'))] * 5
    order = Mock()
    monkeypatch.setattr(agent, "get_order_status", order)
    with pytest.raises(agent.ToolAgentMaxTurnsError):
        agent.ask_with_tools(ticket, "Where is ORD-12345?")
    assert llm.call_count == 5
    order.assert_not_called()
    assert_trace(events(caplog), [LLM] * 5, "error", "max_turns_exceeded")


def test_max_turns_records_every_executed_tool(ticket, external_guards, caplog):
    llm, _ = external_guards
    llm.side_effect = [response(tool())] * 5
    with pytest.raises(agent.ToolAgentMaxTurnsError):
        agent.ask_with_tools(ticket, "Where is ORD-12345?")
    assert_trace(events(caplog), [LLM, "get_order_status"] * 5, "error", "max_turns_exceeded")


def test_sensitive_payloads_absent_from_all_agent_logs(ticket, external_guards, caplog):
    caplog.set_level(logging.INFO)
    llm, _ = external_guards
    llm.side_effect = [response(tool()), response(answer="Private generated response")]
    private_ticket = {**ticket, "customer_name": "Private Customer", "subject": "Private subject",
                      "message": "Private ticket message"}
    agent.ask_with_tools(private_ticket, "Private question ORD-12345", "Private policy text")
    for value in ("Private", "ORD-12345", "PAY-88888", "Laptop", "private-call-id",
                  "order_id", agent.MODEL_NAME):
        assert value not in caplog.text
    logs = events(caplog)
    assert set(logs[0]) == {"event", "trace_id"}
    assert set(logs[1]) == {
        "event", "trace_id", "step_index", "step_type", "operation", "duration_ms",
        "status", "error_category",
    }
    assert set(logs[-1]) == {
        "event", "trace_id", "status", "error_category", "step_count", "total_duration_ms",
    }


@pytest.mark.parametrize("failed_event", [
    "agent_trace_started", "agent_step_completed", "agent_trace_completed",
])
def test_failing_log_handler_does_not_break_agent(ticket, external_guards, failed_event):
    llm, _ = external_guards
    llm.side_effect = [response(tool()), response()]

    class BrokenHandler(logging.Handler):
        def emit(self, record):
            if json.loads(record.getMessage())["event"] == failed_event:
                raise RuntimeError("private handler failure")

    handler = BrokenHandler()
    agent_trace.logger.addHandler(handler)
    try:
        result = agent.ask_with_tools(ticket, "Where is ORD-12345?")
    finally:
        agent_trace.logger.removeHandler(handler)
    assert result["answer"] == "Safe answer"
    assert len(result["verified_facts"]) == 1


def test_logging_failure_does_not_replace_original_exception(monkeypatch, ticket, external_guards):
    llm, _ = external_guards
    original = RuntimeError("private LLM failure")
    llm.side_effect = original
    monkeypatch.setattr(agent_trace.logger, "info", Mock(side_effect=ValueError("logger failed")))
    with pytest.raises(RuntimeError) as raised:
        agent.ask_with_tools(ticket, "Hello")
    assert raised.value is original


def test_concurrent_executions_have_isolated_state(monkeypatch, ticket, caplog):
    barrier = Barrier(2)

    def create(**kwargs):
        barrier.wait(timeout=5)
        return response()

    monkeypatch.setattr(agent.client.chat.completions, "create", create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: agent.ask_with_tools(ticket, "Hello"), range(2)))
    assert results == [{"answer": "Safe answer", "verified_facts": []}] * 2
    logs = events(caplog)
    ids = {e["trace_id"] for e in logs}
    assert len(ids) == 2
    for trace_id in ids:
        assert_trace([e for e in logs if e["trace_id"] == trace_id], [LLM])


def test_durations_use_monotonic_clock(monkeypatch, caplog):
    ticks = iter([10, 11, 11.25, 12])
    monkeypatch.setattr(agent_trace, "perf_counter", lambda: next(ticks))
    with agent_trace.AgentTrace() as trace:
        with trace.step("llm", LLM):
            pass
    logs = events(caplog)
    assert logs[1]["duration_ms"] == 250
    assert logs[-1]["total_duration_ms"] == 2000


@pytest.fixture
def traced_ask(monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app

    retriever = Mock(return_value=None)
    retriever.retrieve.return_value = [(
        {"filename": "private.md", "index": 0, "text": "Private retrieved chunk"}, 0.9,
    )]
    monkeypatch.setattr(app.state, "policy_retriever", retriever, raising=False)
    # No lifespan: the retriever and external calls are explicitly replaced.
    return TestClient(app), retriever


@pytest.mark.parametrize("failed", [False, True])
def test_request_trace_correlates_retrieval_tools_and_llm(
    external_guards, traced_ask, caplog, failed,
):
    from backend.services.rag_service import PolicyRetrievalError

    client, retriever = traced_ask
    llm, _ = external_guards
    if failed:
        retriever.retrieve.side_effect = PolicyRetrievalError("Private retrieval failure")
    llm.side_effect = [response(tool()), response()]
    result = client.post("/tickets/1/ask", json={"question": "Where is ORD-12345?"})
    assert result.status_code == 200
    assert set(result.json()) == {"ticket_id", "answer", "verified_facts"}
    retriever.retrieve.assert_called_once_with("Where is ORD-12345?", top_k=3)
    steps = assert_trace(events(caplog), ["policy_retrieval", LLM, "get_order_status", LLM])
    assert steps[0]["step_type"] == "retrieval"
    assert steps[0]["status"] == ("error" if failed else "success")
    assert steps[0]["error_category"] == ("retrieval_error" if failed else None)
    user_message = llm.call_args.kwargs["messages"][1]["content"]
    assert ("Private retrieved chunk" in user_message) is not failed
    assert "Private" not in caplog.text


def test_request_total_duration_includes_retrieval(monkeypatch, external_guards, traced_ask, caplog):
    client, retriever = traced_ask
    llm, _ = external_guards
    now = [10.0]
    monkeypatch.setattr(agent_trace, "perf_counter", lambda: now[0])

    def retrieve(question, top_k):
        now[0] += 2
        return []

    def create(**kwargs):
        now[0] += 3
        return response()

    retriever.retrieve.side_effect = retrieve
    llm.side_effect = create
    assert client.post("/tickets/1/ask", json={"question": "Hello"}).status_code == 200
    logs = events(caplog)
    steps = assert_trace(logs, ["policy_retrieval", LLM])
    assert [s["duration_ms"] for s in steps] == [2000, 3000]
    assert logs[-1]["total_duration_ms"] == 5000


def test_request_without_retriever_has_no_retrieval_step(
    monkeypatch, external_guards, traced_ask, caplog,
):
    from backend.main import app

    client, retriever = traced_ask
    monkeypatch.setattr(app.state, "policy_retriever", None)
    external_guards[0].side_effect = [response()]
    assert client.post("/tickets/1/ask", json={"question": "Hello"}).status_code == 200
    retriever.retrieve.assert_not_called()
    assert_trace(events(caplog), [LLM])


@pytest.mark.parametrize("failure,category,http_status", [
    ("connection", "llm_unavailable", 503),
    ("authorization", "unauthorized_tool", 502),
    ("max_turns", "max_turns_exceeded", 502),
])
def test_request_owned_trace_finalizes_once_on_agent_failure(
    external_guards, traced_ask, caplog, failure, category, http_status,
):
    client, _ = traced_ask
    llm, _ = external_guards
    if failure == "connection":
        llm.side_effect = APIConnectionError(request=httpx.Request("POST", "http://localhost/v1"))
        operations = ["policy_retrieval", LLM]
    elif failure == "authorization":
        llm.side_effect = [response(tool("kapruka_create_order"))]
        operations = ["policy_retrieval", LLM]
    else:
        llm.side_effect = [response(tool())] * 5
        operations = ["policy_retrieval"] + [LLM, "get_order_status"] * 5
    assert client.post("/tickets/1/ask", json={"question": "Where is ORD-12345?"}).status_code == http_status
    assert_trace(events(caplog), operations, "error", category)


def test_unexpected_retrieval_exception_finalizes_request(external_guards, traced_ask, caplog):
    client, retriever = traced_ask
    retriever.retrieve.side_effect = RuntimeError("Private unexpected failure")
    with pytest.raises(RuntimeError):
        client.post("/tickets/1/ask", json={"question": "Hello"})
    external_guards[0].assert_not_called()
    steps = assert_trace(events(caplog), ["policy_retrieval"], "error", "unexpected_error")
    assert steps[0]["error_category"] == "retrieval_error"
    assert "Private" not in caplog.text


def test_supplied_trace_lifecycle_belongs_to_caller(ticket, external_guards, caplog):
    external_guards[0].side_effect = [response()]
    with agent_trace.AgentTrace() as trace:
        result = agent.ask_with_tools(ticket, "Hello", trace=trace)
        assert result == {"answer": "Safe answer", "verified_facts": []}
        assert [e["event"] for e in events(caplog)] == [
            "agent_trace_started", "agent_step_completed",
        ]
    assert_trace(events(caplog), [LLM])
