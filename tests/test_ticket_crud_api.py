import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.data.database import tickets


@pytest.fixture(autouse=True)
def reset_tickets():
    """Reset the in-memory ticket list to the original two records before each test."""
    tickets.clear()
    tickets.extend(
        [
            {
                "id": 1,
                "customer_name": "Alice Smith",
                "subject": "Login issue",
                "message": "I cannot log into my account.",
                "status": "open",
            },
            {
                "id": 2,
                "customer_name": "Bob Jones",
                "subject": "Billing question",
                "message": "Why was I charged twice this month?",
                "status": "pending",
            },
        ]
    )


def test_get_tickets():
    """GET /tickets/ returns a list of tickets."""
    client = TestClient(app)
    response = client.get("/tickets/")

    assert response.status_code == 200
    tickets = response.json()
    assert isinstance(tickets, list)
    assert len(tickets) >= 1


def test_get_ticket_by_id():
    """GET /tickets/1 returns the ticket with id 1."""
    client = TestClient(app)
    response = client.get("/tickets/1")

    assert response.status_code == 200
    assert response.json()["id"] == 1


def test_get_nonexistent_ticket():
    """GET /tickets/999 returns 404 when the ticket does not exist."""
    client = TestClient(app)
    response = client.get("/tickets/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Ticket not found"}


def test_create_ticket():
    """POST /tickets/ creates a new ticket."""
    client = TestClient(app)
    response = client.post(
        "/tickets/",
        json={
            "customer_name": "Charlie Lee",
            "subject": "Shipping question",
            "message": "When will my order arrive?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["customer_name"] == "Charlie Lee"
    assert body["subject"] == "Shipping question"
    assert body["message"] == "When will my order arrive?"
    assert body["status"] == "open"
    assert "id" in body


def test_create_ticket_missing_message():
    """POST /tickets/ returns 422 when a required field is missing."""
    client = TestClient(app)
    response = client.post(
        "/tickets/",
        json={
            "customer_name": "Charlie Lee",
            "subject": "Shipping question",
        },
    )

    assert response.status_code == 422


def test_create_ticket_whitespace_customer_name_rejected():
    """POST /tickets/ returns 422 and does not create a ticket when customer_name is whitespace-only."""
    client = TestClient(app)
    response = client.post(
        "/tickets/",
        json={
            "customer_name": "   ",
            "subject": "Shipping question",
            "message": "When will my order arrive?",
        },
    )

    assert response.status_code == 422
    assert len(tickets) == 2


def test_create_ticket_whitespace_subject_rejected():
    """POST /tickets/ returns 422 and does not create a ticket when subject is whitespace-only."""
    client = TestClient(app)
    response = client.post(
        "/tickets/",
        json={
            "customer_name": "Charlie Lee",
            "subject": "   ",
            "message": "When will my order arrive?",
        },
    )

    assert response.status_code == 422
    assert len(tickets) == 2


def test_create_ticket_whitespace_message_rejected():
    """POST /tickets/ returns 422 and does not create a ticket when message is whitespace-only."""
    client = TestClient(app)
    response = client.post(
        "/tickets/",
        json={
            "customer_name": "Charlie Lee",
            "subject": "Shipping question",
            "message": "   ",
        },
    )

    assert response.status_code == 422
    assert len(tickets) == 2


def test_update_ticket():
    """PUT /tickets/1 updates the ticket status and keeps other fields."""
    client = TestClient(app)
    response = client.put(
        "/tickets/1",
        json={"status": "resolved"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 1
    assert body["status"] == "resolved"
    assert body["customer_name"] == "Alice Smith"


def test_update_nonexistent_ticket():
    """PUT /tickets/999 returns 404 when the ticket does not exist."""
    client = TestClient(app)
    response = client.put(
        "/tickets/999",
        json={"status": "resolved"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Ticket not found"}


def test_delete_ticket():
    """DELETE /tickets/1 removes the ticket."""
    client = TestClient(app)

    response = client.delete("/tickets/1")

    assert response.status_code == 200
    assert response.json() == {"message": "Ticket deleted"}

    get_response = client.get("/tickets/1")
    assert get_response.status_code == 404


def test_delete_nonexistent_ticket():
    """DELETE /tickets/999 returns 404 when the ticket does not exist."""
    client = TestClient(app)
    response = client.delete("/tickets/999")

    assert response.status_code == 404
    assert response.json() == {"detail": "Ticket not found"}
