"""Basic RAG retrieval experiment.

Reads policy documents from backend/knowledge_base/, chunks them using the
existing approach (500-char max, 100-char overlap), embeds each chunk with
BAAI/bge-small-en-v1.5, and for a set of test questions prints the top-3
most similar chunks ranked by cosine similarity.

Intended as a learning step toward full RAG. No FastAPI, vLLM, vector
database, or persistent storage is used.
"""

from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent  # .../localllm-serve
KB_DIR = BASE_DIR / "backend" / "knowledge_base"
CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP = 100

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

TEST_QUESTIONS = [
    "Can I return my laptop after three weeks?",
    "My package has not arrived. What should I do?",
    "I cannot log in to my account. How can I fix it?",
]

# --- Chunking logic (reused from inspect_chunks.py) ---


def split_into_chunks(text: str, max_chars: int, overlap: int):
    """Yield (chunk_index, chunk_text) tuples from *text*.

    Reuses the same sliding-window approach from inspect_chunks.py:
    always advances, never infinite loops.
    """
    if not text:
        return

    start = 0
    chunk_index = 0
    text_len = len(text)

    while start < text_len:
        end = start + max_chars
        if end > text_len:
            end = text_len

        chunk_text = text[start:end].strip()
        yield chunk_index, chunk_text

        chunk_index += 1

        if end >= text_len:
            start = text_len
        else:
            start = end - overlap
            if start <= 0:
                start = end


# --- Embedding logic ---


def embed_texts(texts, model):
    """Return a list of embedding vectors (numpy arrays) for *texts*."""
    import torch

    embeddings = []
    for text in texts:
        vec = model.encode(text, convert_to_tensor=True)
        embeddings.append(vec.cpu().numpy())
    return embeddings


# --- Main ---


def main():
    if not KB_DIR.is_dir():
        print(f"❌ Knowledge base directory not found: {KB_DIR}")
        return

    md_files = sorted(KB_DIR.glob("*.md"))

    if not md_files:
        print(f"❌ No .md files found in {KB_DIR}")
        return

    # 1️⃣ Chunk all documents, preserving source filename and chunk index
    chunks = []  # list of dicts: {filename, index, text}
    for md_path in md_files:
        filename = md_path.name
        text = md_path.read_text(encoding="utf-8")
        for idx, chunk_text in split_into_chunks(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP):
            if chunk_text:  # skip empty chunks
                chunks.append({
                    "filename": filename,
                    "index": idx,
                    "text": chunk_text,
                })

    if not chunks:
        print("⚠️  No chunks produced from the policy documents.")
        return

    # 2️⃣ Embed all chunks
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = embed_texts(chunk_texts, model)
    print(f"Embedded {len(chunks)} chunks (dimension: {chunk_embeddings[0].shape[0]}).\n")

    # 3️⃣ For each test question, rank chunks by cosine similarity and print top-3
    from sklearn.metrics.pairwise import cosine_similarity

    for question in TEST_QUESTIONS:
        q_emb = model.encode(question, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        sims = cosine_similarity(q_emb, chunk_embeddings).flatten()

        # Get indices of top-3 highest similarities
        top_idx = sims.argsort()[::-1][:3]

        print(f"🔍 Question: \"{question}\"")
        print("-" * 60)

        for rank, i in enumerate(top_idx, start=1):
            c = chunks[i]
            score = sims[i]
            print(f"  Rank {rank}:")
            print(f"    📄 Source: {c['filename']} (chunk {c['index']})")
            print(f"    🎯 Similarity: {score:.4f}")
            # Print chunk text, truncating if very long
            preview = c["text"].strip().replace("\n", " ")[:200]
            if len(c["text"].strip()) > 200:
                preview += "..."
            print(f"    💬 {preview}")
        print()


if __name__ == "__main__":
    main()