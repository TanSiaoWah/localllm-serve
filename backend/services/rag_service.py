"""Policy retrieval with an injected embedding model; no import-time setup."""

from pathlib import Path

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 100


class PolicyRetrievalError(Exception):
    """Raised when policy indexing or retrieval cannot be completed."""


def split_into_chunks(text: str, max_chars: int, overlap: int):
    """Yield bounded chunks, preferring sentence/line breaks and target overlap."""
    import re

    if max_chars <= 0 or overlap < 0:
        raise ValueError("max_chars must be positive and overlap must be non-negative")
    if not text:
        return

    boundaries = [m.end() for m in re.finditer(r"[.!?](?=\s|$)|\n+", text)]
    start = 0
    chunk_index = 0
    text_len = len(text)
    previous_end = 0

    while start < text_len:
        while start < text_len and text[start].isspace():
            start += 1
        if start >= text_len:
            break

        limit = min(start + max_chars, text_len)
        end = limit
        if limit < text_len:
            # Require new source text, rather than emitting only overlap again.
            candidates = [b for b in boundaries if max(start, previous_end) < b <= limit]
            if candidates:
                end = candidates[-1]
            else:
                # With no sentence/line break, split at whitespace if possible.
                spaces = [
                    i for i in range(max(start, previous_end) + 1, limit + 1)
                    if text[i].isspace()
                ]
                if spaces:
                    end = spaces[-1]

        chunk_text = text[start:end].strip()
        yield chunk_index, chunk_text
        chunk_index += 1
        previous_end = end

        if end >= text_len:
            break

        # Align overlap to a boundary; never move backwards or skip source text.
        candidates = [b for b in boundaries if start < b < end] if overlap else []
        start = min(candidates, key=lambda b: (abs(b - (end - overlap)), -b)) if candidates else end


def _cosine_similarity(a, b):
    import numpy as np

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


class PolicyRetriever:
    """Build one in-memory policy index explicitly, then reuse it for questions."""

    def __init__(self, embedding_model, kb_dir: Path):
        self.embedding_model = embedding_model
        self.kb_dir = Path(kb_dir)
        self._chunks = []
        self._embeddings = []
        self._initialized = False

    def initialize(self):
        """Read and embed documents once. Failed initialization can be retried."""
        if self._initialized:
            return

        chunks = []
        try:
            if not self.kb_dir.is_dir():
                raise PolicyRetrievalError(
                    f"Knowledge base directory not found: {self.kb_dir}"
                )
            for path in sorted(self.kb_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                for index, chunk_text in split_into_chunks(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP):
                    chunks.append({
                        "filename": path.name,
                        "index": index,
                        "text": chunk_text,
                    })
        except OSError as exc:
            raise PolicyRetrievalError("Could not read policy documents") from exc

        if not chunks:
            raise PolicyRetrievalError("Knowledge base contains no non-empty Markdown chunks")

        try:
            embeddings = self.embedding_model.encode(
                [chunk["text"] for chunk in chunks], convert_to_tensor=False
            )
            embeddings = [list(vector) for vector in embeddings]
            if len(embeddings) != len(chunks):
                raise ValueError("Expected one embedding per document chunk")
        except Exception as exc:
            raise PolicyRetrievalError("Could not embed policy documents") from exc

        # Publish the complete index only after all setup has succeeded.
        self._chunks = chunks
        self._embeddings = embeddings
        self._initialized = True

    def retrieve(self, question, top_k=3):
        """Return ranked (chunk dictionary, cosine score) pairs."""
        if not self._initialized:
            raise PolicyRetrievalError("Policy retriever has not been initialized")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0:
            return []

        try:
            question_embedding = self.embedding_model.encode(
                [question], convert_to_tensor=False
            )[0]
            scores = [
                _cosine_similarity(question_embedding, vector)
                for vector in self._embeddings
            ]
        except Exception as exc:
            raise PolicyRetrievalError("Could not retrieve policy evidence") from exc

        indices = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]
        return [(self._chunks[index], scores[index]) for index in indices]
