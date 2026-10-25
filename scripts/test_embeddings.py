"""Minimal sentence-transformers embedding experiment.

- Loads a small pretrained sentence-BERT model.
- Encodes three example texts.
- Prints the embedding dimension.
- Calculates cosine similarity between a question and two policy
  passages (one relevant, one unrelated) using scikit-learn.
- Uses the installed library's similarity utilities rather than
  implementing cosine similarity from scratch.

Intended as a learning step toward RAG. No chunking, retrieval,
persistent storage, vector database, or LLM integration is performed.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch


# The three texts from the task description.
QUESTION: str = "Can I send my laptop back after three weeks?"
RELEVANT_POLICY: str = "Customers may return most products within 30 days of delivery."
UNRELATED_POLICY: str = "International shipping typically takes 10 to 15 business days."


def main() -> None:
    # Load a lightweight, ready-to-use English SBERT model.
    # BAAI/bge-small-en-v1.5 produces 384-dimensional embeddings.
    model: SentenceTransformer = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Encode the three sentences into dense vector embeddings.
    # convert_to_tensor=True returns PyTorch tensors.
    question_embedding = model.encode(QUESTION, convert_to_tensor=True)
    relevant_embedding = model.encode(RELEVANT_POLICY, convert_to_tensor=True)
    unrelated_embedding = model.encode(UNRELATED_POLICY, convert_to_tensor=True)

    # Print the embedding dimension (length of the vector).
    dim: int = question_embedding.size(0)
    print(f"Embedding dimension: {dim}")

    # Calculate cosine similarity using scikit-learn.
    # sklearn expects 2D arrays of shape (n_samples, n_features).
    # Unsqueeze the 1D embeddings to 2D: (1, dim).
    sim_relevant = cosine_similarity(
        question_embedding.cpu().numpy().reshape(1, -1),
        relevant_embedding.cpu().numpy().reshape(1, -1),
    ).item()
    sim_unrelated = cosine_similarity(
        question_embedding.cpu().numpy().reshape(1, -1),
        unrelated_embedding.cpu().numpy().reshape(1, -1),
    ).item()

    # Label the scores clearly.
    print(f"Similarity (question ↔ relevant policy):   {sim_relevant:.4f}")
    print(f"Similarity (question ↔ unrelated policy): {sim_unrelated:.4f}")


if __name__ == "__main__":
    main()