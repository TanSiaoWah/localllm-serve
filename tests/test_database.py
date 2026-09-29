from sqlalchemy import create_engine

from backend.data.database import DATABASE_URL, SessionLocal, engine


def test_engine_builds_from_database_url():
    """The SQLAlchemy engine can be created from the configured DATABASE_URL."""
    # Rebuild an engine from the same URL to prove the configuration is valid.
    # This does not open a connection, so the Supabase database is untouched.
    new_engine = create_engine(DATABASE_URL)

    assert DATABASE_URL is not None
    assert new_engine.url is not None
    assert new_engine.url.get_backend_name() == "postgresql"


def test_module_engine_uses_postgresql():
    """The engine exposed by the data layer targets PostgreSQL."""
    assert engine.url.get_backend_name() == "postgresql"


def test_session_factory_creates_session():
    """The session factory can hand out a (lazy, not-yet-connected) session."""
    session = SessionLocal()
    try:
        assert session is not None
    finally:
        session.close()