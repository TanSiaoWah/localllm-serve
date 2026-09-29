import logging

from backend.tools.payment_tools import get_payment_status
import backend.data.payment_repository as payment_repository


def test_valid_existing_payment(db_storage):
    """A valid existing payment returns its details (from the isolated DB)."""
    result = get_payment_status("PAY-88888")
    assert result["status"] == "captured"
    assert result["amount"] == 1299.0
    assert result["currency"] == "MYR"


def test_invalid_payment_id_type(db_storage):
    """An ID that does not start with PAY- is rejected."""
    result = get_payment_status("ORD-12345")
    assert result == {"error": "Invalid payment ID"}


def test_nonexistent_payment(db_storage):
    """A correctly formatted but nonexistent payment returns not found."""
    result = get_payment_status("PAY-99999")
    assert result == {"error": "Payment not found"}


def test_payment_database_failure_returns_controlled_error(db_storage, monkeypatch):
    """A database/SQLAlchemy error returns a controlled result, never the raw error."""

    def boom(session, payment_id):
        raise RuntimeError("connection failed: SELECT * FROM payments WHERE ...")

    monkeypatch.setattr(payment_repository, "get_payment", boom)

    result = get_payment_status("PAY-88888")

    assert result == {"error": "Payment service temporarily unavailable"}


def test_payment_tool_logs_success(db_storage, caplog):
    """A successful payment lookup logs the invocation and the found outcome."""
    caplog.set_level(logging.INFO, logger="backend.tools.payment_tools")

    get_payment_status("PAY-88888")

    assert "Payment tool invoked: payment_id=PAY-88888" in caplog.text
    assert "Payment found: payment_id=PAY-88888" in caplog.text


def test_payment_tool_logs_not_found(db_storage, caplog):
    """A missing payment logs a not-found outcome."""
    caplog.set_level(logging.INFO, logger="backend.tools.payment_tools")

    get_payment_status("PAY-99999")

    assert "Payment not found: payment_id=PAY-99999" in caplog.text


def test_payment_tool_logs_service_unavailable_without_leaking_error(
    db_storage, monkeypatch, caplog
):
    """A DB failure logs the outcome without leaking the raw exception text."""

    def boom(session, payment_id):
        raise RuntimeError("postgresql://user:secret@host/db connection refused")

    monkeypatch.setattr(payment_repository, "get_payment", boom)
    caplog.set_level(logging.INFO, logger="backend.tools.payment_tools")

    result = get_payment_status("PAY-88888")

    assert result == {"error": "Payment service temporarily unavailable"}
    assert "Payment lookup failed: payment_id=PAY-88888" in caplog.text
    # The raw exception / connection details must not appear in the logs.
    assert "postgresql://" not in caplog.text
    assert "connection refused" not in caplog.text