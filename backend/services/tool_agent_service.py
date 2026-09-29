import json

from openai import APIConnectionError, OpenAI

from backend.tools.order_tools import get_order_status
from backend.tools.payment_tools import get_payment_status
from backend.tools.schemas import SUPPORT_TOOLS

# Connect to the local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Model served by vLLM
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"

class ToolAgentUnavailableError(Exception):
    """Raised when the local vLLM server cannot be reached."""


class InvalidToolCallError(Exception):
    """Raised when the LLM returns an invalid tool call."""


class UnknownToolError(Exception):
    """Raised when the LLM requests a tool the backend does not support."""


class ToolAgentMaxTurnsError(Exception):
    """Raised when the tool agent exceeds its maximum number of LLM turns."""


# System prompt explaining how to use ticket and tool information
SYSTEM_PROMPT = (
    "You are a customer support agent.\n"
    "- The customer ticket contains information reported by the customer.\n"
    "- Tool results contain verified backend information.\n"
    "- Answer general support questions without tools when verified business data is not required.\n"
    "- For factual business data such as order status, use the tools instead of guessing.\n"
    "- If a required order or payment ID is missing, ask the customer for it instead of inventing one.\n"
    "- Never claim to have checked account, order, or payment information unless a tool returned it.\n"
    "- For general issues such as login problems, give useful troubleshooting guidance without claiming account-specific facts.\n"
    "- Never invent facts that are not present in the ticket or in tool results.\n"
    "- When appropriate, distinguish verified information from interpretation.\n"
    "- Order IDs start with ORD-.\n"
    "- Payment IDs start with PAY-.\n"
    "- get_order_status accepts order IDs.\n"
    "- get_payment_status accepts payment IDs.\n"
    "- Never pass an order ID to get_payment_status.\n"
    "- If asked about payment for an order and only an order ID is known, first call get_order_status.\n"
    "- Use the returned payment_id to call get_payment_status.\n"
    "- In your final answer, state business facts ONLY when they are present in the ticket or in the exact tool results you received.\n"
    "- Do NOT invent email notifications, refund policies, fulfillment consequences, processing timelines, support actions, account status, or any other business facts that are not in the ticket or tool results.\n"
    "- Do not recommend or state that no further action is required unless that recommendation is explicitly supported by the ticket data or actual tool results.\n"
    "- When answering with current business data, report the exact tool result rather than assumptions or general model knowledge.\n"
    "- If you describe a likely consequence or interpretation based on verified facts, clearly mark it as an interpretation or possibility, not as a verified business fact."
)


def ask_with_tools(ticket: dict, question: str) -> dict:
    """Send a ticket and question to Qwen, execute tools, and return the answer with verified facts."""
    # Build the user message with the ticket context and the agent question
    user_content = (
        f"Customer support ticket:\n"
        f"Subject: {ticket['subject']}\n"
        f"Message: {ticket['message']}\n\n"
        f"Support agent question:\n"
        f"{question}"
    )

    # Collect the verified backend facts returned by tools
    verified_facts = []

    # Start the conversation with the system and user messages
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    # Simple agent loop with a maximum of 5 LLM turns
    for _ in range(5):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=SUPPORT_TOOLS,
                tool_choice="auto",
            )
        except APIConnectionError as exc:
            raise ToolAgentUnavailableError(
                "Local vLLM server is unavailable"
            ) from exc

        assistant_message = response.choices[0].message

        # If no tool was requested, return the answer with the verified facts
        if not assistant_message.tool_calls:
            return {
                "answer": assistant_message.content,
                "verified_facts": verified_facts,
            }

        # Keep the assistant message with its tool calls in the conversation
        messages.append(
            {
                "role": "assistant",
                "tool_calls": assistant_message.tool_calls,
            }
        )

        # Execute each requested tool and append the results
        for tool_call in assistant_message.tool_calls:
            print(f"Tool requested: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")

            # Parse the arguments and execute the requested tool
            try:
                arguments = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "get_order_status":
                    result = get_order_status(arguments["order_id"])
                    print(f"Tool result: {result}")
                elif tool_call.function.name == "get_payment_status":
                    result = get_payment_status(arguments["payment_id"])
                    print(f"Tool result: {result}")
                else:
                    raise UnknownToolError(
                        "LLM requested an unknown tool"
                    )
            except (json.JSONDecodeError, KeyError) as exc:
                raise InvalidToolCallError(
                    "LLM returned an invalid tool call"
                ) from exc

            # Record the verified backend fact
            verified_facts.append(
                {
                    "tool": tool_call.function.name,
                    "result": result,
                }
            )

            # Append a tool message for this result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    # If the loop finishes without a plain answer, stop
    raise ToolAgentMaxTurnsError(
        "Tool agent exceeded maximum turns"
    )
