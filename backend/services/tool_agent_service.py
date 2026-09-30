import json
import logging
import re

from openai import APIConnectionError, OpenAI

from backend.tools.order_tools import get_order_status
from backend.tools.payment_tools import get_payment_status
from backend.tools.schemas import ORDER_STATUS_TOOL, PAYMENT_STATUS_TOOL, SUPPORT_TOOLS

logger = logging.getLogger(__name__)

# Maximum number of LLM turns the tool agent will take.
MAX_TURNS = 5

# Connect to the local vLLM server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

# Model served by vLLM
MODEL_NAME = "Qwen/Qwen3-8B-AWQ"

# Explicit business identifiers in the customer's original question.
ORDER_ID_PATTERN = re.compile(r"ORD-[A-Za-z0-9]+")
PAYMENT_ID_PATTERN = re.compile(r"PAY-[A-Za-z0-9]+")

# Controlled tool errors when the model requests an identifier that does not
# match the one the customer explicitly provided.
ORDER_ID_MISMATCH_ERROR = (
    "The requested order ID does not match the order ID provided by the customer."
)
PAYMENT_ID_MISMATCH_ERROR = (
    "The requested payment ID does not match the payment ID provided by the customer."
)

# Payment-related intent in the customer's original question.
PAYMENT_INTENT_PATTERN = re.compile(r"payment|charge|transaction|paid", re.IGNORECASE)


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
    "- When the customer explicitly provides an order or payment ID, use exactly that identifier when calling the corresponding tool; never substitute, reuse, or infer a different identifier.\n"
    "- If asked about payment for an order and only an order ID is known, first call get_order_status.\n"
    "- Use the returned payment_id to call get_payment_status.\n"
    "- Use only the tools necessary to answer the customer's actual question.\n"
    "- Do not call unrelated tools just because related business information exists.\n"
    "- If the customer asks only for order status, call get_order_status only and stop tool use after obtaining the order status.\n"
    "- Call get_payment_status only when the customer explicitly asks about payment, a charge, a transaction, or payment status, or when answering the customer's question genuinely requires payment information.\n"
    "- When the customer asks about payment for an order, call get_order_status first to obtain the payment_id, then call get_payment_status.\n"
    "- In your final answer, state business facts ONLY when they are present in the ticket or in the exact tool results you received.\n"
    "- Do NOT invent email notifications, refund policies, fulfillment consequences, processing timelines, support actions, account status, or any other business facts that are not in the ticket or tool results.\n"
    "- Do not recommend or state that no further action is required unless that recommendation is explicitly supported by the ticket data or actual tool results.\n"
    "- When answering with current business data, report the exact tool result rather than assumptions or general model knowledge.\n"
    "- If a tool returns an error result, do not invent the requested business information and do not claim the operation succeeded; clearly tell the customer that the requested information could not be verified.\n"
    "- If the tool error is a \"not found\" error, explain that the requested record could not be found.\n"
    "- If the tool error is a \"service temporarily unavailable\" error, explain that the information cannot currently be verified because the relevant service is unavailable.\n"
    "- Never repeat or expose SQLAlchemy errors, PostgreSQL errors, connection strings, or any internal implementation details.\n"
    "- If you describe a likely consequence or interpretation based on verified facts, clearly mark it as an interpretation or possibility, not as a verified business fact."
)


def _build_tools_for_request(
    customer_order_ids: list[str],
    customer_payment_ids: list[str],
    payment_intent: bool,
) -> list[dict]:
    """Construct the tool list the model is allowed to see for this request.

    1. Customer provides an order ID + payment intent:
       expose [ORDER_STATUS_TOOL, PAYMENT_STATUS_TOOL]
    2. Customer provides an order ID without payment intent:
       expose [ORDER_STATUS_TOOL]
    3. Customer provides a payment ID:
       expose [PAYMENT_STATUS_TOOL]
    4. No relevant order/payment ID:
       expose no business tools, even if payment_intent is True.
    """
    if customer_order_ids and payment_intent:
        return [ORDER_STATUS_TOOL, PAYMENT_STATUS_TOOL]
    if customer_order_ids:
        return [ORDER_STATUS_TOOL]
    if customer_payment_ids:
        return [PAYMENT_STATUS_TOOL]
    return []


def ask_with_tools(ticket: dict, question: str) -> dict:
    """Send a ticket and question to Qwen, execute tools, and return the answer with verified facts."""
    # Explicit identifiers the customer provided in the original question.
    # When present, model-requested tool calls must use exactly these values.
    customer_order_ids = ORDER_ID_PATTERN.findall(question)
    customer_payment_ids = PAYMENT_ID_PATTERN.findall(question)
    # Whether the customer explicitly asked about payment-related information.
    payment_intent = PAYMENT_INTENT_PATTERN.search(question) is not None
    # Per-request guard: mismatched identifier calls already rejected once.
    rejected_mismatch_calls = set()

    # Determine which tools are exposed to Qwen for this specific request.
    tools_for_request = _build_tools_for_request(
        customer_order_ids, customer_payment_ids, payment_intent
    )

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

    # Simple agent loop with a maximum of MAX_TURNS LLM turns
    for _ in range(MAX_TURNS):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools_for_request or None,
                tool_choice="auto" if tools_for_request else None,
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
            tool_name = tool_call.function.name
            logger.info("Tool call requested by model: %s", tool_name)

            # Parse the arguments and execute the requested tool
            try:
                arguments = json.loads(tool_call.function.arguments)
                if tool_name == "get_order_status":
                    order_id = arguments["order_id"]
                    # Reject an ID the customer never provided; never substitute.
                    if customer_order_ids and order_id not in customer_order_ids:
                        mismatch_key = (tool_name, order_id)
                        if mismatch_key in rejected_mismatch_calls:
                            logger.warning(
                                "Tool call rejected again: repeated mismatched "
                                "order identifier"
                            )
                        else:
                            rejected_mismatch_calls.add(mismatch_key)
                            logger.warning(
                                "Tool call rejected: requested order ID does not "
                                "match the customer question"
                            )
                        result = {"error": ORDER_ID_MISMATCH_ERROR}
                    else:
                        result = get_order_status(order_id)
                elif tool_name == "get_payment_status":
                    payment_id = arguments["payment_id"]
                    if customer_payment_ids and payment_id not in customer_payment_ids:
                        mismatch_key = (tool_name, payment_id)
                        if mismatch_key in rejected_mismatch_calls:
                            logger.warning(
                                "Tool call rejected again: repeated mismatched "
                                "payment identifier"
                            )
                        else:
                            rejected_mismatch_calls.add(mismatch_key)
                            logger.warning(
                                "Tool call rejected: requested payment ID does "
                                "not match the customer question"
                            )
                        result = {"error": PAYMENT_ID_MISMATCH_ERROR}
                    else:
                        result = get_payment_status(payment_id)
                else:
                    raise UnknownToolError(
                        "LLM requested an unknown tool"
                    )
            except (json.JSONDecodeError, KeyError) as exc:
                raise InvalidToolCallError(
                    "LLM returned an invalid tool call"
                ) from exc

            # Log the outcome without logging the full result/record.
            if isinstance(result, dict) and "error" in result:
                logger.warning("Tool %s returned an error", tool_name)
            else:
                logger.info("Tool %s completed successfully", tool_name)

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
    logger.warning("Tool agent reached maximum turns (%d)", MAX_TURNS)
    raise ToolAgentMaxTurnsError(
        "Tool agent exceeded maximum turns"
    )
