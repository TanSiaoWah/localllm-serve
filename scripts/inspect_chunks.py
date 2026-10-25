"""Inspect and chunk the knowledge-base Markdown files.

Reads .md files from backend/knowledge_base/, splits them into overlapping
character chunks (max 500 chars, 100-char overlap), and prints each chunk
with its source filename, index, and character count.

Intended as a learning step for RAG document chunking -- no embeddings,
vector search, or LLM integration is performed.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # .../localllm-serve
KB_DIR = BASE_DIR / "backend" / "knowledge_base"
MAX_CHARS = 500
OVERLAP = 100


def split_into_chunks(text: str, max_chars: int, overlap: int):
    """Yield (chunk_index, chunk_text) tuples from *text*.

    The chunking always advances because the start position moves past the
    previous chunk's end by at least 1 character when there is enough text
    remaining.  When fewer than *overlap* characters remain, the overlap is
    reduced so the final chunk ends at the string end and the loop terminates.
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

        chunk_text = text[start:end]
        yield chunk_index, chunk_text.strip()

        chunk_index += 1

        # Advance: move start to the next position.
        # If we consumed the entire remaining text, advance past the end
        # so the loop terminates.  Otherwise step forward by (max_chars -
        # overlap), ensuring at least 1 character of progress.
        if end >= text_len:
            start = text_len  # consumed everything -> exit loop
        else:
            start = end - overlap
            # Guard against start not advancing (possible when overlap > max_chars
            # or near the end of the string).  Force at least +1 progress.
            if start <= 0:
                start = end


def main():
    if not KB_DIR.is_dir():
        print(f"Knowledge base directory not found: {KB_DIR}")
        return

    md_files = sorted(KB_DIR.glob("*.md"))

    if not md_files:
        print("No .md files found in", KB_DIR)
        return

    for md_path in md_files:
        filename = md_path.name
        text = md_path.read_text(encoding="utf-8")

        for chunk_index, chunk_text in split_into_chunks(text, MAX_CHARS, OVERLAP):
            char_count = len(chunk_text)
            print(f"FILE: {filename}")
            print(f"  CHUNK {chunk_index}: {char_count} chars")
            print(f"  {chunk_text}")
            print()


if __name__ == "__main__":
    main()