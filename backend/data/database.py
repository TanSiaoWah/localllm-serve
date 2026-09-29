import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables from backend/.env (holds DATABASE_URL).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Connection string for the Supabase PostgreSQL database. It is kept in
# backend/.env (gitignored) and is never logged or printed.
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL is None:
    raise RuntimeError(
        "DATABASE_URL is not set. Add a DATABASE_URL entry to backend/.env"
    )

# SQLAlchemy engine and session factory for the Supabase database.
# Creating the engine does not open a connection; connections are created
# lazily the first time a query runs.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
