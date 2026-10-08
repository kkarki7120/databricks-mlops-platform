import numpy as np
import pytest

from src.rag.chunking import chunk_text
from src.rag.retrieval import build_prompt, top_k


def test_short_text_is_one_chunk():
    assert chunk_text("Para one.\n\nPara two.", max_chars=500) == ["Para one.\n\nPara two."]


def test_chunks_respect_max_size_and_are_not_empty():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 300 for c in chunks)


def test_long_paragraph_uses_overlapping_window():
    text = "x" * 1000
    chunks = chunk_text(text, max_chars=400, overlap=100)
    assert all(len(c) <= 400 for c in chunks)
    assert sum(len(c) for c in chunks) > 1000  # overlap means some characters repeat


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=100, overlap=100)


def test_top_k_returns_most_similar_first():
    matrix = np.array([[1, 0], [0, 1], [0.9, 0.1]])
    idx, scores = top_k([1, 0], matrix, k=2)
    assert list(idx) == [0, 2]
    assert scores[0] >= scores[1]


def test_top_k_dimension_mismatch():
    with pytest.raises(ValueError):
        top_k([1, 0, 0], np.ones((3, 2)))


def test_prompt_contains_question_and_sources():
    prompt = build_prompt("Refund period?", [{"source": "refund_policy", "text": "30 days."}])
    assert "Refund period?" in prompt and "[refund_policy]" in prompt and "30 days." in prompt
