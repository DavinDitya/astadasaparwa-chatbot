# utils.py
import re
from typing import List
from math import ceil

def clean_text(text: str) -> str:
    """
    Basic cleaning: normalize whitespace, remove weird control chars but keep punctuation.
    Keep case (don't lowercase automatically) — but you can lowercase if you want.
    """
    if not text:
        return ""
    # replace multiple newlines/spaces with single space
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.strip()
    # optionally remove non-printable chars
    text = re.sub(r'[^\x00-\x7F\u00A0-\uFFFF]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """
    Chunk text into pieces of up to max_chars with an overlap (characters).
    This uses a simple character-based windowing for predictable behavior.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end >= len(text):
            break
        # move start forward but keep overlap
        start = end - overlap
    return chunks
