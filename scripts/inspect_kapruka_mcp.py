"""Discover the Kapruka product-search schema and run one read-only search."""

import asyncio
import json
import sys
import traceback

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SERVER_URL = "https://mcp.kapruka.com/mcp"
TOOL_NAME = "kapruka_search_products"


async def main():
    # Open the HTTP transport, then start the session's message handling.
    async with streamable_http_client(SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(
            read_stream, write_stream, read_timeout_seconds=30
        ) as session:
            # Negotiate MCP capabilities before requesting tool definitions.
            await session.initialize()
            result = await session.list_tools()
            tool = next((tool for tool in result.tools if tool.name == TOOL_NAME), None)
            if tool is None:
                raise RuntimeError(f"Tool discovery did not find {TOOL_NAME!r}.")

            print(f"Tool name: {tool.name}")
            print(f"Description: {tool.description or '(not provided)'}")
            print("Input schema:")
            print(json.dumps(tool.input_schema, indent=2))

            result = await session.call_tool(
                "kapruka_search_products",
                arguments={
                    "params": {
                        "q": "laptop",
                        "limit": 3,
                        "currency": "LKR",
                        "response_format": "json",
                    }
                },
            )
            # SDK 2.3.0 exposes the MCP wire field isError as is_error.
            if result.is_error:
                raise RuntimeError(
                    "Kapruka product search returned a tool error:\n"
                    + result.model_dump_json(indent=2)
                )

            print("\nProduct search result:")
            # Preserve all returned fields and original JSON text content.
            print(result.model_dump_json(indent=2))
        # Context managers close the session and transport, including on errors.


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print("Kapruka MCP connection, discovery, or product search failed.", file=sys.stderr)
        # Include underlying causes, including nested async transport errors.
        traceback.print_exception(exc)
        sys.exit(1)
