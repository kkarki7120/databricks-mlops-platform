"""Text chunking for RAG.

Strategy: split on paragraphs (keeps related sentences together), pack paragraphs
into chunks up to `max_chars`, and fall back to a sliding window with `overlap`
for paragraphs that are too long on their own.
"""


def chunk_text(text: str, max_chars: int = 500, overlap: int = 100) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap < max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(para) <= max_chars:
            current = para
        else:
            step = max_chars - overlap
            for start in range(0, len(para), step):
                piece = para[start:start + max_chars]
                chunks.append(piece)
                if start + max_chars >= len(para):
                    break

    if current:
        chunks.append(current)
    return chunks
