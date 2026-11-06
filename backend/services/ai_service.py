import json
import os

from openai import APIConnectionError, OpenAI, Timeout
from pydantic import ValidationError

from backend.models import TicketAnalysis


client = OpenAI(
    base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key="EMPTY",
    timeout=Timeout(connect=5.0, read=120.0, write=30.0, pool=5.0),
    max_retries=0,
)

MODEL_NAME = "Qwen/Qwen3-8B-AWQ"


class InvalidAIResponseError(Exception):
    """Raised when the LLM returns invalid structured output."""


class AIServiceUnavailableError(Exception):
    """Raised when the local vLLM server cannot be reached."""


def analyze_ticket(ticket: dict) -> TicketAnalysis:
    """Send a ticket to the local LLM and return a validated analysis."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Here is a customer support ticket:\n"
                        f"Subject: {ticket['subject']}\n"
                        f"Message: {ticket['message']}\n\n"
                        f"Return only JSON with exactly these fields:\n"
                        f'{{"issue": "short description", '
                        f'"category": "account | billing | shipping | refund | technical | other", '
                        f'"urgency": "low | medium | high"}}'
                    ),
                }
            ],
        )
    except APIConnectionError as exc:
        raise AIServiceUnavailableError(
            "Local vLLM server is unavailable"
        ) from exc

    content = response.choices[0].message.content

    if content is None:
        raise InvalidAIResponseError(
            "LLM returned invalid structured output"
        )

    try:
        parsed = json.loads(content)
        return TicketAnalysis(**parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InvalidAIResponseError(
            "LLM returned invalid structured output"
        ) from exc
