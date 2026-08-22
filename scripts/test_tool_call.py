from backend.services.tool_agent_service import ask_with_tools


ticket = {
    "subject": "Order and payment question",
    "message": "I want to check my order and payment.",
}


print(
    ask_with_tools(
        ticket,
        "Check whether the payment for order ORD-12345 was successful."
    )
)