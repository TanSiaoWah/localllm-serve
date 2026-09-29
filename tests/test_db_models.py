from backend.data.db_models import Order, Payment, Ticket


def _column_map(model):
    return {column.name: column for column in model.__table__.columns}


def test_ticket_is_mapped_to_support_schema():
    """The Ticket table lives in the \"support\" PostgreSQL schema."""
    assert Ticket.__table__.schema == "support"


def test_ticket_table_name_is_tickets():
    """The Ticket table is named \"tickets\"."""
    assert Ticket.__tablename__ == "tickets"


def test_expected_columns_exist():
    """The expected business columns are present on the model."""
    columns = _column_map(Ticket)

    expected = [
        "id",
        "customer_name",
        "subject",
        "message",
        "status",
        "created_at",
    ]
    for name in expected:
        assert name in columns, f"missing column {name!r}"

    # The four payload columns are required, matching the database NOT NULL.
    for name in ["customer_name", "subject", "message", "status", "created_at"]:
        assert columns[name].nullable is False


def test_id_is_primary_key():
    """The \"id\" column is the table's primary key."""
    assert list(Ticket.__table__.primary_key.columns.keys()) == ["id"]


def test_payment_is_mapped_to_support_schema():
    """The Payment table lives in the \"support\" PostgreSQL schema."""
    assert Payment.__table__.schema == "support"


def test_payment_table_name_is_payments():
    """The Payment table is named \"payments\"."""
    assert Payment.__tablename__ == "payments"


def test_payment_expected_columns_exist():
    """The expected business columns are present on the Payment model."""
    columns = _column_map(Payment)

    expected = ["payment_id", "status", "amount", "currency"]
    for name in expected:
        assert name in columns, f"missing column {name!r}"

    for name in ["status", "amount", "currency"]:
        assert columns[name].nullable is False


def test_payment_primary_key():
    """The \"payment_id\" column is the Payment primary key."""
    assert list(Payment.__table__.primary_key.columns.keys()) == ["payment_id"]


def test_order_is_mapped_to_support_schema():
    """The Order table lives in the \"support\" PostgreSQL schema."""
    assert Order.__table__.schema == "support"


def test_order_table_name_is_orders():
    """The Order table is named \"orders\"."""
    assert Order.__tablename__ == "orders"


def test_order_expected_columns_exist():
    """The expected business columns are present on the Order model."""
    columns = _column_map(Order)

    expected = ["order_id", "product", "status", "payment_id"]
    for name in expected:
        assert name in columns, f"missing column {name!r}"

    for name in ["product", "status", "payment_id"]:
        assert columns[name].nullable is False


def test_order_primary_key():
    """The \"order_id\" column is the Order primary key."""
    assert list(Order.__table__.primary_key.columns.keys()) == ["order_id"]


def test_order_payment_id_foreign_key():
    """Order.payment_id references support.payments.payment_id."""
    payment_id = _column_map(Order)["payment_id"]
    fk = list(payment_id.foreign_keys)
    assert len(fk) == 1
    assert fk[0].target_fullname == "support.payments.payment_id"


def test_order_payment_id_unique_and_not_null():
    """Order.payment_id is unique and non-nullable."""
    payment_id = _column_map(Order)["payment_id"]
    assert payment_id.unique is True
    assert payment_id.nullable is False