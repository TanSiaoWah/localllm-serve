from backend.tools.order_tools import get_order_status


def test_valid_existing_order():
    """A valid existing order returns its details."""
    result = get_order_status("ORD-12345")
    assert result["product"] == "Laptop"
    assert result["status"] == "shipped"
    assert result["payment_id"] == "PAY-88888"


def test_invalid_order_id_type():
    """An ID that does not start with ORD- is rejected."""
    result = get_order_status("PAY-88888")
    assert result == {"error": "Invalid order ID"}


def test_nonexistent_order():
    """A correctly formatted but nonexistent order returns not found."""
    result = get_order_status("ORD-99999")
    assert result == {"error": "Order not found"}