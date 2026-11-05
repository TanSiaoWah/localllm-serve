"""Per-execution agent timing; emit metadata only, never payloads or exceptions."""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

logger = logging.getLogger("backend.agent_trace")


def configure_trace_logging():
    """Enable terminal JSON traces at app startup without configuring root logs."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # The dedicated handler owns output; ancestor handlers must not emit it again.
    logger.propagate = False


@dataclass
class TraceStep:
    error_category: str | None = None


class AgentTrace:
    """Stream ordered events without retaining steps or sharing request state."""

    def __init__(self):
        self.trace_id = uuid4().hex
        self._started = perf_counter()
        self._step_index = 0
        self.error_category = None

    def _emit(self, event: str, **fields):
        try:
            logger.info(json.dumps({"event": event, "trace_id": self.trace_id, **fields}))
        except Exception:
            # Includes custom handlers/formatters that raise. No fallback log:
            # observability must not interrupt execution or leak error details.
            pass

    def __enter__(self):
        self._emit("agent_trace_started")
        return self

    def __exit__(self, exc_type, exc, traceback):
        category = self.error_category
        if exc_type is not None and category is None:
            category = "unexpected_error"
        self._emit(
            "agent_trace_completed",
            status="error" if category else "success",
            error_category=category,
            step_count=self._step_index,
            total_duration_ms=(perf_counter() - self._started) * 1000,
        )
        # Never suppress the original exception.
        return False

    @contextmanager
    def step(self, step_type: str, operation: str):
        self._step_index += 1
        index = self._step_index
        started = perf_counter()
        step = TraceStep()
        try:
            yield step
        except BaseException:
            step.error_category = f"{step_type}_error"
            raise
        finally:
            self._emit(
                "agent_step_completed",
                step_index=index,
                step_type=step_type,
                operation=operation,
                duration_ms=(perf_counter() - started) * 1000,
                status="error" if step.error_category else "success",
                error_category=step.error_category,
            )
