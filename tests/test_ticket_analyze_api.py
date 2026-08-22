from fastapi.testclient import TestClient

from backend.main import app
import backend.services.ai_service as ai_service
from backend.models import TicketAnalysis
from backend.services.ai_service import (
    AIServiceUnavailableError,
    InvalidAIResponseError,
)


def test_analyze_ticket_with_mocked_ai(monkeypatch):
    """POST /tickets/1/analyze returns the mocked analysis."""

    def fake_analyze_ticket(ticket):
        return TicketAnalysis(
            issue="Cannot log into account",
            category="account",
            urgency="high",
        )

    monkeypatch.setattr(ai_service, "analyze_ticket", fake_analyze_ticket)

    client = TestClient(app)
    response = client.post("/tickets/1/analyze")

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == 1
    assert body["analysis"]["issue"] == "Cannot log into account"
    assert body["analysis"]["category"] == "account"
    assert body["analysis"]["urgency"] == "high"


def test_analyze_ticket_invalid_ai_output(monkeypatch):
    """POST /tickets/1/analyze returns 502 when the LLM returns invalid output."""

    def fake_analyze_ticket(ticket):
        raise InvalidAIResponseError("LLM returned invalid structured output")

    monkeypatch.setattr(ai_service, "analyze_ticket", fake_analyze_ticket)

    client = TestClient(app)
    response = client.post("/tickets/1/analyze")

    assert response.status_code == 502
    assert response.json() == {"detail": "LLM returned invalid structured output"}


def test_analyze_ticket_service_unavailable(monkeypatch):
    """POST /tickets/1/analyze returns 503 when the LLM service is unavailable."""

    def fake_analyze_ticket(ticket):
        raise AIServiceUnavailableError("Local vLLM server is unavailable")

    monkeypatch.setattr(ai_service, "analyze_ticket", fake_analyze_ticket)

    client = TestClient(app)
    response = client.post("/tickets/1/analyze")

    assert response.status_code == 503
    assert response.json() == {"detail": "Local LLM service is unavailable"}
