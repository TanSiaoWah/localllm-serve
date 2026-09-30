# OpenAI tool schemas describing available backend tools.
# Individual constants allow the service layer to construct a per-request
# tool list so that irrelevant tools are never exposed to the model.

ORDER_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": "Get the status of an order by its order ID",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up",
                },
            },
            "required": ["order_id"],
        },
    },
}

PAYMENT_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_payment_status",
        "description": "Get the status of a payment by its payment ID",
        "parameters": {
            "type": "object",
            "properties": {
                "payment_id": {
                    "type": "string",
                    "description": "The payment ID to look up",
                },
            },
            "required": ["payment_id"],
        },
    },
}

SUPPORT_TOOLS = [ORDER_STATUS_TOOL, PAYMENT_STATUS_TOOL]
