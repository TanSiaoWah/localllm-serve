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