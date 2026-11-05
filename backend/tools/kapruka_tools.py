"""Read-only Kapruka product search, independent of the support agent and DB."""

import asyncio
import json

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_URL = "https://mcp.kapruka.com/mcp"
SUPPORTED_CURRENCIES = {"LKR", "USD", "GBP", "AUD", "EUR"}


class KaprukaValidationError(ValueError):
    """Invalid local search arguments; no MCP request was made."""


class KaprukaMCPError(Exception):
    """Kapruka could not provide a valid product-search response."""


def _parse_search_result(result) -> dict:
    # SDK 2.3.0 uses snake_case for isError and structuredContent.
    if result.is_error:
        raise KaprukaMCPError("Kapruka product search reported an error.")

    payload = result.structured_content
    if payload is None:
        # Only text blocks can contain the requested JSON response.
        payload = "\n".join(
            block.text for block in result.content if block.type == "text"
        )

    try:
        if isinstance(payload, str):
            payload = json.loads(payload)
        # Kapruka wraps its JSON response in structured_content.result.
        if isinstance(payload, dict) and "results" not in payload:
            payload = payload.get("result")
            if isinstance(payload, str):
                payload = json.loads(payload)
    except (ValueError, RecursionError):
        raise KaprukaMCPError("Kapruka returned invalid product JSON.") from None

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise KaprukaMCPError("Kapruka returned an unexpected product response.")

    # Preserve product fields as external data; never execute descriptions.
    data = {
        "source": "kapruka",
        "data_type": "product_catalog",
        "results": payload["results"],
    }
    for key in ("next_cursor", "applied_filters"):
        if key in payload:
            data[key] = payload[key]
    return data


async def _search_kapruka_products(query: str, limit: int, currency: str) -> dict:
    try:
        # Bound HTTP connect time to 10s and other HTTP operations to 30s.
        async with httpx2.AsyncClient(
            timeout=httpx2.Timeout(30.0, connect=10.0)
        ) as http_client:
            async with streamable_http_client(
                SERVER_URL, http_client=http_client
            ) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream, write_stream, read_timeout_seconds=30
                ) as session:
                    await session.initialize()
                    # This is the only permitted tool; no generic dispatch.
                    result = await session.call_tool(
                        "kapruka_search_products",
                        arguments={
                            "params": {
                                "q": query,
                                "limit": limit,
                                "currency": currency,
                                "response_format": "json",
                            }
                        },
                    )
                    data = _parse_search_result(result)
        # All resources close before returning, including on error paths.
        return data
    except KaprukaMCPError:
        raise
    except Exception:
        # Includes protocol, transport, timeout and async cleanup failures.
        raise KaprukaMCPError("Kapruka product search is temporarily unavailable.") from None


def search_kapruka_products(query: str, limit: int = 3, currency: str = "LKR") -> dict:
    """Search product terms only; callers must not pass private customer data.

    Call from synchronous code, such as the existing agent's worker thread.
    Async callers must offload this function to a worker thread.
    """
    if not isinstance(query, str):
        raise KaprukaValidationError("query must be a string.")
    query = query.strip()
    if not 3 <= len(query) <= 200:
        raise KaprukaValidationError("query must contain 3 to 200 characters.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10:
        raise KaprukaValidationError("limit must be an integer between 1 and 10.")
    if not isinstance(currency, str) or currency not in SUPPORTED_CURRENCIES:
        raise KaprukaValidationError("currency must be LKR, USD, GBP, AUD, or EUR.")

    # Avoid creating an unawaited coroutine inside an already-running loop.
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise KaprukaMCPError("Call product search from a synchronous worker thread.")

    return asyncio.run(_search_kapruka_products(query, limit, currency))
