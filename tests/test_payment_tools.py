from backend.tools.payment_tools import get_payment_status


def test_valid_existing_payment():
    """A valid existing payment returns its details."""
    result = get_payment_status("PAY-88888")
    assert result["status"] == "captured"
    assert result["amount"] == 1299.0
    assert result["currency"] == "MYR"


def test_invalid_payment_id_type():
    """An ID that does not start with PAY- is rejected."""
    result = get_payment_status("ORD-12345")
    assert result == {"error": "Invalid payment ID"}


def test_nonexistent_payment():
    """A correctly formatted but nonexistent payment returns not found."""
    result = get_payment_status("PAY-99999")
    assert result == {"error": "Payment not found"}