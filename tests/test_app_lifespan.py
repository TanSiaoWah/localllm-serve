import builtins
import importlib.util
import json
import logging
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.main as main
import backend.services.tool_agent_service as tool_agent_service
from backend.services import agent_trace


@pytest.fixture(autouse=True)
def isolated_trace_logging(monkeypatch):
    # Startup config persists for a worker's lifetime; isolate it between tests.
    monkeypatch.setattr(agent_trace.logger, "handlers", [])
    monkeypatch.setattr(agent_trace.logger, "level", logging.NOTSET)
    monkeypatch.setattr(agent_trace.logger, "propagate", True)


def test_trace_info_visible_with_root_warning_and_no_root_changes(monkeypatch, capsys):
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.WARNING)
    original_handlers = list(root.handlers)
    uvicorn_logger = logging.getLogger("uvicorn.error")
    original_uvicorn = (uvicorn_logger.level, list(uvicorn_logger.handlers))
    assert not agent_trace.logger.isEnabledFor(logging.INFO)

    agent_trace.configure_trace_logging()
    with agent_trace.AgentTrace():
        pass

    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert [r["event"] for r in records] == ["agent_trace_started", "agent_trace_completed"]
    assert records[0]["trace_id"] == records[1]["trace_id"]
    assert captured.out == ""
    assert root.level == logging.WARNING
    assert root.handlers == original_handlers
    assert (uvicorn_logger.level, uvicorn_logger.handlers) == original_uvicorn


def test_trace_setup_is_idempotent_and_does_not_propagate(monkeypatch, capsys):
    # A root handler must not receive a second copy, even if it accepts INFO.
    from unittest.mock import Mock

    ancestor_handler = logging.StreamHandler()
    ancestor_handler.emit = Mock()
    monkeypatch.setattr(logging.getLogger(), "handlers", [ancestor_handler])
    agent_trace.configure_trace_logging()
    handler = agent_trace.logger.handlers[0]
    agent_trace.configure_trace_logging()
    assert agent_trace.logger.handlers == [handler]

    with agent_trace.AgentTrace():
        pass

    assert len(capsys.readouterr().err.splitlines()) == 2
    ancestor_handler.emit.assert_not_called()


def test_trace_setup_reuses_existing_handler(capsys):
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    agent_trace.logger.addHandler(handler)
    agent_trace.configure_trace_logging()
    assert agent_trace.logger.handlers == [handler]
    with agent_trace.AgentTrace():
        pass
    assert len(capsys.readouterr().err.splitlines()) == 2


def test_lifespan_enables_json_traces_for_ask(monkeypatch, capsys):
    from unittest.mock import Mock

    retriever = Mock()
    retriever.retrieve.return_value = []
    monkeypatch.setattr(main, "create_policy_retriever", lambda: retriever)
    message = types.SimpleNamespace(content="Safe answer", tool_calls=None)
    llm = Mock(return_value=types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)]))
    monkeypatch.setattr(tool_agent_service.client.chat.completions, "create", llm)

    # Two startup cycles must still produce exactly one copy of each event.
    for _ in range(2):
        with TestClient(main.app) as client:
            result = client.post("/tickets/1/ask", json={"question": "Private customer question"})
            assert result.status_code == 200
            assert result.json() == {"ticket_id": 1, "answer": "Safe answer", "verified_facts": []}

    captured = capsys.readouterr()
    records = [json.loads(line) for line in captured.err.splitlines()]
    assert [r["event"] for r in records] == [
        "agent_trace_started", "agent_step_completed", "agent_step_completed", "agent_trace_completed",
    ] * 2
    assert len({r["trace_id"] for r in records}) == 2
    assert "Private customer question" not in captured.err
    assert "Safe answer" not in captured.err
    assert len(agent_trace.logger.handlers) == 1


def test_cpu_factory_and_initialization_once_per_lifespan(monkeypatch):
    model_calls = []
    retrievers = []
    model = object()

    def fake_model(name, *, device):
        model_calls.append((name, device))
        assert main.app.state.policy_retriever is None
        return model

    class FakeRetriever:
        def __init__(self, embedding_model, kb_dir):
            assert embedding_model is model
            assert kb_dir.is_absolute()
            assert kb_dir == Path(main.__file__).resolve().parent / "knowledge_base"
            self.initializations = 0
            retrievers.append(self)

        def initialize(self):
            assert main.app.state.policy_retriever is None
            self.initializations += 1

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = fake_model
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(main, "PolicyRetriever", FakeRetriever)

    client = TestClient(main.app)
    assert model_calls == []
    with client:
        retriever = main.app.state.policy_retriever
        assert retriever is retrievers[0]
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/tickets/").status_code == 200
        assert main.app.state.policy_retriever is retriever
        assert retriever.initializations == 1
        assert model_calls == [("BAAI/bge-small-en-v1.5", "cpu")]
    assert main.app.state.policy_retriever is None


@pytest.mark.parametrize("failure_stage", ["factory", "initialize"])
def test_rag_failure_logs_and_preserves_routes(monkeypatch, caplog, failure_stage):
    calls = []
    class FakeRetriever:
        def initialize(self):
            calls.append("initialize")
            assert main.app.state.policy_retriever is None
            raise RuntimeError("simulated index failure")

    def factory():
        calls.append("factory")
        assert main.app.state.policy_retriever is None
        if failure_stage == "factory":
            raise RuntimeError("simulated model loading failure")
        return FakeRetriever()

    def fake_agent(ticket, question, policy_context="", *, trace=None):
        return {"answer": "Existing agent still works.", "verified_facts": []}

    monkeypatch.setattr(main, "create_policy_retriever", factory)
    monkeypatch.setattr(tool_agent_service, "ask_with_tools", fake_agent)
    # A prior state value must be cleared before either setup stage runs.
    monkeypatch.setattr(main.app.state, "policy_retriever", object(), raising=False)
    caplog.set_level(logging.ERROR, logger="backend.main")

    with TestClient(main.app) as client:
        assert main.app.state.policy_retriever is None
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/tickets/").status_code == 200
        response = client.post("/tickets/1/ask", json={"question": "Help me."})
        assert response.status_code == 200
        assert response.json() == {
            "ticket_id": 1, "answer": "Existing agent still works.", "verified_facts": []
        }
        assert calls == (["factory"] if failure_stage == "factory" else ["factory", "initialize"])
    assert "continuing without RAG" in caplog.text
    assert any(record.exc_info for record in caplog.records if record.name == "backend.main")


@pytest.mark.parametrize("failure_stage", [None, "factory", "initialize"])
def test_readiness_reports_rag_startup_status_without_reinitializing(monkeypatch, failure_stage):
    from unittest.mock import Mock

    retriever = Mock()
    factory = Mock(return_value=retriever)
    failure = RuntimeError("Private model initialization details")
    if failure_stage == "factory":
        factory.side_effect = failure
    elif failure_stage == "initialize":
        retriever.initialize.side_effect = failure
    monkeypatch.setattr(main, "create_policy_retriever", factory)
    # Startup must discard any retriever left by an earlier lifespan.
    monkeypatch.setattr(main.app.state, "policy_retriever", object(), raising=False)

    with TestClient(main.app) as client:
        for _ in range(2):
            response = client.get("/ready")
            assert response.status_code == 200
            assert response.json() == {
                "status": "ready" if failure_stage is None else "degraded"
            }
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
        factory.assert_called_once_with()
        if failure_stage == "factory":
            retriever.initialize.assert_not_called()
        else:
            retriever.initialize.assert_called_once_with()
        retriever.retrieve.assert_not_called()


def test_distinct_applications_have_distinct_retrievers(monkeypatch):
    retrievers = []
    class FakeRetriever:
        def __init__(self):
            self.initializations = 0

        def initialize(self):
            self.initializations += 1

    def factory():
        retriever = FakeRetriever()
        retrievers.append(retriever)
        return retriever

    monkeypatch.setattr(main, "create_policy_retriever", factory)
    first_app = FastAPI(lifespan=main.lifespan)
    second_app = FastAPI(lifespan=main.lifespan)
    with TestClient(first_app), TestClient(second_app):
        assert first_app.state.policy_retriever is retrievers[0]
        assert second_app.state.policy_retriever is retrievers[1]
        assert retrievers[0] is not retrievers[1]
        assert [retriever.initializations for retriever in retrievers] == [1, 1]
    assert first_app.state.policy_retriever is None
    assert second_app.state.policy_retriever is None


def test_main_import_does_not_import_or_initialize_embedding_model(monkeypatch):
    spec = importlib.util.find_spec("backend.main")
    code = spec.loader.get_code(spec.name)
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"sentence_transformers", "torch"}:
            raise AssertionError("Embedding model imported before lifespan startup")
        return original_import(name, *args, **kwargs)

    def fail_initialize(self):
        raise AssertionError("Retriever initialized before lifespan startup")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(main.PolicyRetriever, "initialize", fail_initialize)
    namespace = {"__name__": "isolated_backend_main", "__file__": main.__file__}
    exec(code, namespace)
    assert isinstance(namespace["app"], FastAPI)
