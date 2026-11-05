"""Offline tests: HTTP, MCP transport, and MCP session are all replaced."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import backend.tools.kapruka_tools as kapruka


def _response(payload=None, *, text="", is_error=False):
    return SimpleNamespace(
        is_error=is_error,
        structured_content=payload,
        content=[SimpleNamespace(type="text", text=text)],
    )


@pytest.fixture
def catalog():
    return {
        "results": [{
            "id": "laptop-1",
            "name": "Example laptop",
            "price": 123456.75,
            "currency": "LKR",
            "in_stock": True,
            "url": "https://example.com/laptop-1",
            "description": "External product text, not instructions.",
        }],
        "next_cursor": "page-2",
        "applied_filters": {"q": "laptop"},
    }


@pytest.fixture(autouse=True)
def mcp_mock(monkeypatch, catalog):
    """Prevent network access even when an individual test forgets to mock it."""
    state = SimpleNamespace(
        events=[], result=_response({"result": json.dumps(catalog)}),
        connection_error=None, initialize_error=None, call_error=None,
    )

    async def initialize():
        state.events.append("initialize")
        if state.initialize_error:
            raise state.initialize_error

    async def call_tool(*args, **kwargs):
        state.events.append("call")
        if state.call_error:
            raise state.call_error
        return state.result

    state.session = SimpleNamespace(
        initialize=AsyncMock(side_effect=initialize),
        call_tool=AsyncMock(side_effect=call_tool),
    )
    state.http = object()
    state.read_stream, state.write_stream = object(), object()

    @asynccontextmanager
    async def http_client(**kwargs):
        state.events.append("http open")
        try:
            yield state.http
        finally:
            state.events.append("http close")

    @asynccontextmanager
    async def transport(*args, **kwargs):
        state.events.append("transport open")
        if state.connection_error:
            raise state.connection_error
        try:
            yield state.read_stream, state.write_stream
        finally:
            state.events.append("transport close")

    @asynccontextmanager
    async def session(*args, **kwargs):
        state.events.append("session open")
        try:
            yield state.session
        finally:
            state.events.append("session close")

    state.http_factory = Mock(side_effect=http_client)
    state.transport = Mock(side_effect=transport)
    state.session_factory = Mock(side_effect=session)
    monkeypatch.setattr(kapruka.httpx2, "AsyncClient", state.http_factory)
    monkeypatch.setattr(kapruka, "streamable_http_client", state.transport)
    monkeypatch.setattr(kapruka, "ClientSession", state.session_factory)
    return state


@pytest.mark.parametrize("currency", ["LKR", "USD", "GBP", "AUD", "EUR"])
@pytest.mark.parametrize("query,limit", [("abc", 1), ("x" * 200, 10)])
def test_valid_inputs_and_supported_currencies(mcp_mock, query, limit, currency):
    kapruka.search_kapruka_products(query, limit, currency)
    mcp_mock.session.call_tool.assert_awaited_once_with(
        "kapruka_search_products",
        arguments={"params": {
            "q": query, "limit": limit, "currency": currency,
            "response_format": "json",
        }},
    )


@pytest.mark.parametrize("kwargs", [
    {"query": "ab"}, {"query": ""}, {"query": "   "},
    {"query": "x" * 201}, {"query": None}, {"query": 123},
    {"query": ["laptop"]},
    *[{"query": "laptop", "limit": value}
      for value in (True, False, 0, -1, 11, 1.5, "3", None)],
    *[{"query": "laptop", "currency": value}
      for value in ("MYR", "usd", "", None, [])],
])
def test_invalid_arguments_rejected_before_connection(mcp_mock, kwargs):
    with pytest.raises(kapruka.KaprukaValidationError):
        kapruka.search_kapruka_products(**kwargs)
    mcp_mock.http_factory.assert_not_called()
    mcp_mock.transport.assert_not_called()
    mcp_mock.session.call_tool.assert_not_awaited()


def test_search_contract_trimming_lifecycle_and_timeouts(mcp_mock, catalog):
    result = kapruka.search_kapruka_products("  laptop\n")
    assert result == {"source": "kapruka", "data_type": "product_catalog", **catalog}
    mcp_mock.transport.assert_called_once_with(
        "https://mcp.kapruka.com/mcp", http_client=mcp_mock.http
    )
    mcp_mock.session_factory.assert_called_once_with(
        mcp_mock.read_stream, mcp_mock.write_stream, read_timeout_seconds=30
    )
    timeout = mcp_mock.http_factory.call_args.kwargs["timeout"]
    assert timeout.connect == 10
    assert timeout.read == timeout.write == timeout.pool == 30
    mcp_mock.session.initialize.assert_awaited_once_with()
    mcp_mock.session.call_tool.assert_awaited_once_with(
        "kapruka_search_products",
        arguments={"params": {
            "q": "laptop", "limit": 3, "currency": "LKR",
            "response_format": "json",
        }},
    )
    assert mcp_mock.events == [
        "http open", "transport open", "session open", "initialize", "call",
        "session close", "transport close", "http close",
    ]


@pytest.mark.parametrize("kind", ["wrapped", "direct", "string", "text"])
def test_response_formats_preserve_products_and_labels(catalog, kind):
    if kind == "wrapped":
        response = _response({"result": json.dumps(catalog)})
    elif kind == "direct":
        response = _response(catalog)
    elif kind == "string":
        response = _response(json.dumps(catalog))
    else:
        response = _response(text=json.dumps(catalog))
    assert kapruka._parse_search_result(response) == {
        "source": "kapruka", "data_type": "product_catalog", **catalog,
    }


def test_structured_content_takes_precedence(catalog):
    response = _response({"result": json.dumps(catalog)}, text="invalid fallback")
    assert kapruka._parse_search_result(response)["results"] == catalog["results"]


def test_empty_results_are_valid():
    assert kapruka._parse_search_result(_response({"results": []})) == {
        "source": "kapruka", "data_type": "product_catalog", "results": [],
    }


@pytest.mark.parametrize("response", [
    _response({"results": []}, is_error=True),
    _response({"result": "not JSON"}), _response(text="not JSON"),
    _response({}), _response({"other": []}),
    _response({"results": None}), _response({"results": {}}),
    _response({"results": "products"}), _response(text="[]"),
    _response(text="null"),
])
def test_invalid_response_raises_controlled_error(response):
    with pytest.raises(kapruka.KaprukaMCPError):
        kapruka._parse_search_result(response)


@pytest.mark.parametrize("stage", ["connection", "initialize", "call"])
@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
def test_external_failures_are_controlled_and_cleanup_occurs(mcp_mock, stage, error_type):
    setattr(mcp_mock, f"{stage}_error", error_type("private internal details"))
    with pytest.raises(kapruka.KaprukaMCPError) as exc:
        kapruka.search_kapruka_products("laptop")
    assert str(exc.value) == "Kapruka product search is temporarily unavailable."
    assert mcp_mock.events[-1] == "http close"
    if stage == "connection":
        mcp_mock.session_factory.assert_not_called()
    else:
        assert mcp_mock.events[-3:] == [
            "session close", "transport close", "http close",
        ]
    if stage != "call":
        mcp_mock.session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("response", [
    _response(is_error=True), _response({"result": "malformed JSON"}),
])
def test_tool_and_parsing_errors_also_close_resources(mcp_mock, response):
    mcp_mock.result = response
    with pytest.raises(kapruka.KaprukaMCPError):
        kapruka.search_kapruka_products("laptop")
    assert mcp_mock.events[-3:] == [
        "session close", "transport close", "http close",
    ]


@pytest.mark.parametrize("kwargs", [
    {"tool_name": "kapruka_create_order"},
    {"tool_name": "kapruka_track_order"},
    {"server_url": "https://example.com/mcp"},
])
def test_tool_and_server_overrides_are_not_accepted(mcp_mock, kwargs):
    with pytest.raises(TypeError):
        kapruka.search_kapruka_products("laptop", **kwargs)
    mcp_mock.transport.assert_not_called()
    mcp_mock.session.call_tool.assert_not_awaited()


@pytest.mark.parametrize("query", ["kapruka_create_order", "kapruka_track_order"])
def test_tool_names_in_query_cannot_change_dispatch(mcp_mock, query):
    kapruka.search_kapruka_products(query)
    assert mcp_mock.session.call_tool.await_count == 1
    call = mcp_mock.session.call_tool.await_args
    assert call.args == ("kapruka_search_products",)
    assert call.kwargs["arguments"]["params"]["q"] == query


def test_running_event_loop_is_rejected_before_connection(mcp_mock):
    async def run():
        with pytest.raises(kapruka.KaprukaMCPError, match="worker thread"):
            kapruka.search_kapruka_products("laptop")

    asyncio.run(run())
    mcp_mock.http_factory.assert_not_called()
    mcp_mock.transport.assert_not_called()
