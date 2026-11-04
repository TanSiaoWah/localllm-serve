import builtins
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import backend.main as main
import backend.services.tool_agent_service as tool_agent_service


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

    def fake_agent(ticket, question):
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
