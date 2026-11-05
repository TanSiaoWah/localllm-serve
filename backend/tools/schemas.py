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

# Reviewed external catalog tool; not yet included in agent tool selection.
KAPRUKA_SEARCH_PRODUCTS_TOOL = {
    "type": "function",
    "function": {
        "name": "kapruka_search_products",
        "description": (
            "Search Kapruka's external Sri Lankan product catalog. "
            "Prices are in the requested supported currency, not MYR. "
            "Kapruka products are separate from the demo company's orders, "
            "payments, and return policies; those policies do not establish "
            "Kapruka return eligibility. Use product search terms only, "
            "never private customer, ticket, order, or payment data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product search terms, without private customer data",
                    "minLength": 3,
                    "maxLength": 200,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of products to return",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
                "currency": {
                    "type": "string",
                    "description": "Requested price currency; MYR is not supported",
                    "enum": ["LKR", "USD", "GBP", "AUD", "EUR"],
                    "default": "LKR",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
