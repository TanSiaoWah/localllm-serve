import builtins
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from backend.services.rag_service import (
    PolicyRetrievalError,
    PolicyRetriever,
    split_into_chunks,
)


class FakeEmbeddingModel:
    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.calls = []
        self.fail = False

    def encode(self, texts, convert_to_tensor=False):
        assert convert_to_tensor is False
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        return [self.vectors.get(text, [1.0, 0.0]) for text in texts]


@pytest.mark.parametrize("text", [
    "A short policy sentence.",
    " ".join(f"Rule {i} requires verification before a refund." for i in range(80)),
    " ".join(f"condition{i}" for i in range(200)),
    "# Policy\n\n" + "\n".join(f"- Rule {i}: retain all original accessories." for i in range(40)),
], ids=["short", "sentences", "unpunctuated", "markdown"])
def test_chunk_size_metadata_indexes_and_source_coverage(text):
    chunks = list(split_into_chunks(text, 500, 100))
    assert [index for index, _ in chunks] == list(range(len(chunks)))
    covered = set()
    previous_start = -1
    previous_end = 0
    for _, chunk in chunks:
        assert chunk and len(chunk) <= 500
        start = text.find(chunk, previous_start + 1)
        assert start >= 0
        end = start + len(chunk)
        assert end > previous_end
        covered.update(range(start, end))
        previous_start, previous_end = start, end
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())


def test_sentence_and_markdown_boundaries():
    sentences = [f"Rule {i}: " + "a" * 200 + "." for i in range(6)]
    chunks = list(split_into_chunks(" ".join(sentences), 500, 100))
    assert all(chunk.startswith("Rule ") and chunk.endswith(".") for _, chunk in chunks)
    bullets = "\n".join(f"- Condition {i}: " + "b" * 90 for i in range(12))
    chunks = list(split_into_chunks(bullets, 500, 100))
    assert all(chunk.startswith("- Condition ") and chunk.endswith("b" * 90) for _, chunk in chunks)


def test_unbroken_word_and_empty_text():
    text = "x" * 1201
    chunks = list(split_into_chunks(text, 500, 100))
    assert [len(chunk) for _, chunk in chunks] == [500, 500, 201]
    assert "".join(chunk for _, chunk in chunks) == text
    assert list(split_into_chunks("", 500, 100)) == []
    assert list(split_into_chunks(" \n\t" * 500, 500, 100)) == []


def test_metadata_ranking_top_k_and_cached_documents(tmp_path):
    # Filenames are deliberately created out of order; vector magnitude must
    # not outweigh direction when using cosine similarity.
    for filename, text in [("b.md", "Shipping policy."), ("a.md", "Return policy."), ("c.md", "Login policy.")]:
        (tmp_path / filename).write_text(text, encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("Not Markdown", encoding="utf-8")
    model = FakeEmbeddingModel({
        "Return policy.": [10, 0], "Shipping policy.": [1, 1],
        "Login policy.": [0, 2], "returns?": [2, 0], "login?": [0, 1],
    })
    retriever = PolicyRetriever(model, tmp_path)
    assert model.calls == []
    retriever.initialize()
    retriever.initialize()
    assert model.calls == [["Return policy.", "Shipping policy.", "Login policy."]]

    results = retriever.retrieve("returns?")
    assert [chunk["filename"] for chunk, _ in results] == ["a.md", "b.md", "c.md"]
    assert [score for _, score in results] == pytest.approx([1, 2 ** -0.5, 0])
    assert results[0][0] == {"filename": "a.md", "index": 0, "text": "Return policy."}
    assert retriever.retrieve("login?", top_k=1)[0][0]["filename"] == "c.md"
    assert len(retriever.retrieve("returns?", top_k=10)) == 3
    assert model.calls[1:] == [["returns?"], ["login?"], ["returns?"]]
    assert retriever.retrieve("unused", top_k=0) == []
    assert len(model.calls) == 4
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("unused", top_k=-1)


def test_multi_chunk_metadata(tmp_path):
    text = "\n".join(f"- Policy {i}: preserve these conditions." for i in range(40))
    (tmp_path / "policy.md").write_text(text, encoding="utf-8")
    retriever = PolicyRetriever(FakeEmbeddingModel(), tmp_path)
    retriever.initialize()
    results = retriever.retrieve("question", top_k=100)
    # Equal scores keep the original document/chunk order.
    assert [chunk for chunk, _ in results] == [
        {"filename": "policy.md", "index": index, "text": chunk}
        for index, chunk in split_into_chunks(text, 500, 100)
    ]


@pytest.mark.parametrize("contents", [None, "", " \n\t"])
def test_empty_knowledge_base(tmp_path, contents):
    if contents is not None:
        (tmp_path / "empty.md").write_text(contents, encoding="utf-8")
    model = FakeEmbeddingModel()
    with pytest.raises(PolicyRetrievalError, match="no non-empty Markdown"):
        PolicyRetriever(model, tmp_path).initialize()
    assert model.calls == []


def test_missing_knowledge_base_and_uninitialized_retrieval(tmp_path):
    retriever = PolicyRetriever(FakeEmbeddingModel(), tmp_path / "missing")
    with pytest.raises(PolicyRetrievalError, match="has not been initialized"):
        retriever.retrieve("question")
    with pytest.raises(PolicyRetrievalError, match="directory not found"):
        retriever.initialize()


def test_embedding_failure_and_retry(tmp_path):
    (tmp_path / "policy.md").write_text("Policy text.", encoding="utf-8")
    model = FakeEmbeddingModel()
    retriever = PolicyRetriever(model, tmp_path)
    model.fail = True
    with pytest.raises(PolicyRetrievalError, match="embed policy documents") as error:
        retriever.initialize()
    assert isinstance(error.value.__cause__, RuntimeError)
    with pytest.raises(PolicyRetrievalError, match="has not been initialized"):
        retriever.retrieve("question")
    model.fail = False
    retriever.initialize()
    model.fail = True
    with pytest.raises(PolicyRetrievalError, match="retrieve policy evidence"):
        retriever.retrieve("question")
    model.fail = False
    assert retriever.retrieve("question")[0][1] == pytest.approx(1)
    assert model.calls == [["Policy text."], ["Policy text."], ["question"], ["question"]]


def test_document_read_failure(tmp_path, monkeypatch):
    (tmp_path / "policy.md").write_text("Policy text.", encoding="utf-8")
    def fail_read(*args, **kwargs):
        raise OSError("simulated read failure")
    monkeypatch.setattr(Path, "read_text", fail_read)
    with pytest.raises(PolicyRetrievalError, match="read policy documents"):
        PolicyRetriever(FakeEmbeddingModel(), tmp_path).initialize()


def test_import_and_construction_have_no_model_or_file_side_effects(monkeypatch):
    # Obtain module code before guarding I/O, then execute a fresh module body.
    spec = importlib.util.find_spec("backend.services.rag_service")
    code = spec.loader.get_code(spec.name)
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"sentence_transformers", "torch", "openai"}:
            raise AssertionError(f"Unexpected model/client import: {name}")
        return original_import(name, *args, **kwargs)
    def fail_io(*args, **kwargs):
        raise AssertionError("Unexpected file access")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", fail_io)
    for method in ("read_text", "glob", "is_dir"):
        monkeypatch.setattr(Path, method, fail_io)
    namespace = {"__name__": "isolated_rag_service"}
    exec(code, namespace)
    model = FakeEmbeddingModel()
    namespace["PolicyRetriever"](model, Path("unused"))
    assert model.calls == []


def test_standalone_script_delegates_without_generation(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "test_rag_answer.py"
    model = FakeEmbeddingModel()
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = lambda name: model
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    # Avoid the script's sys.path insertion leaking into other tests.
    monkeypatch.setattr(sys, "path", list(sys.path))
    kb_dir = tmp_path / "backend" / "knowledge_base"
    kb_dir.mkdir(parents=True)
    (kb_dir / "policy.md").write_text("Policy text.", encoding="utf-8")
    namespace = {
        "__name__": "standalone_rag_test",
        "__file__": str(tmp_path / "scripts" / "test_rag_answer.py"),
    }
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    assert len(model.calls) == 1
    assert namespace["split_into_chunks"] is split_into_chunks
    results = namespace["_retrieve_chunks"]("test question", top_k=2)
    assert results[0][0] == {"filename": "policy.md", "index": 0, "text": "Policy text."}
    assert model.calls[1:] == [["test question"]]
