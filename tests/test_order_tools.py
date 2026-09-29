import logging

from backend.tools.order_tools import get_order_status
import backend.data.order_repository as order_repository


def test_valid_existing_order(db_storage):
    """A valid existing order returns its details (from the isolated DB)."""
    result = get_order_status("ORD-12345")
    assert result["product"] == "Laptop"
    assert result["status"] == "shipped"
    assert result["payment_id"] == "PAY-88888"


def test_invalid_order_id_type(db_storage):
    """An ID that does not start with ORD- is rejected."""
    result = get_order_status("PAY-88888")
    assert result == {"error": "Invalid order ID"}


def test_nonexistent_order(db_storage):
    """A correctly formatted but nonexistent order returns not found."""
    result = get_order_status("ORD-99999")
    assert result == {"error": "Order not found"}


def test_order_database_failure_returns_controlled_error(db_storage, monkeypatch):
    """A database/SQLAlchemy error returns a controlled result, never the raw error."""

    def boom(session, order_id):
        raise RuntimeError("connection failed: SELECT * FROM orders WHERE ...")

    monkeypatch.setattr(order_repository, "get_order", boom)

    result = get_order_status("ORD-12345")

    assert result == {"error": "Order service temporarily unavailable"}


def test_order_tool_logs_success(db_storage, caplog):
    """A successful order lookup logs the invocation and the found outcome."""
    caplog.set_level(logging.INFO, logger="backend.tools.order_tools")

    get_order_status("ORD-12345")

    assert "Order tool invoked: order_id=ORD-12345" in caplog.text
    assert "Order found: order_id=ORD-12345" in caplog.text


def test_order_tool_logs_not_found(db_storage, caplog):
    """A missing order logs a not-found outcome."""
    caplog.set_level(logging.INFO, logger="backend.tools.order_tools")

    get_order_status("ORD-99999")

    assert "Order not found: order_id=ORD-99999" in caplog.text


def test_order_tool_logs_service_unavailable_without_leaking_error(
    db_storage, monkeypatch, caplog
):
    """A DB failure logs the outcome without leaking the raw exception text."""

    def boom(session, order_id):
        raise RuntimeError("postgresql://user:secret@host/db connection refused")

    monkeypatch.setattr(order_repository, "get_order", boom)
    caplog.set_level(logging.INFO, logger="backend.tools.order_tools")

    result = get_order_status("ORD-12345")

    assert result == {"error": "Order service temporarily unavailable"}
    assert "Order lookup failed: order_id=ORD-12345" in caplog.text
    # The raw exception / connection details must not appear in the logs.
    assert "postgresql://" not in caplog.text
    assert "connection refused" not in caplog.text