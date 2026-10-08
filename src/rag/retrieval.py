"""Retrieval and prompt construction for RAG.

`top_k` is a brute-force cosine-similarity search: fine for a few thousand chunks.
In production this is replaced by a vector index (Databricks AI Search), but the
maths is identical, which makes it easy to explain and to unit test.
"""
import numpy as np

SYSTEM_PROMPT = (
    "You are a customer support assistant. Answer ONLY using the context provided. "
    "If the answer is not in the context, say \"I don't know based on the available documents.\" "
    "Cite the source document names you used in square brackets, e.g. [refund_policy]."
)


def top_k(query_vec, matrix, k: int = 3):
    """Return (indices, scores) of the k rows in `matrix` most similar to `query_vec`."""
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    if m.ndim != 2 or m.shape[1] != q.shape[0]:
        raise ValueError(f"Dimension mismatch: matrix {m.shape}, query {q.shape}")
    sims = (m @ q) / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-10)
    k = min(k, len(sims))
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def build_prompt(question: str, contexts: list[dict]) -> str:
    """contexts: list of {"source": str, "text": str}."""
    blocks = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in contexts)
    return f"Context:\n{blocks}\n\nQuestion: {question}\nAnswer:"
