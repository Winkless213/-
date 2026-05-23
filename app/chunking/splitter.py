import re

from app.chunking.models import Chunk


def _has_headings(text: str) -> bool:
    return bool(re.search(r"^#{1,6}\s+", text, re.MULTILINE))


def _split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split markdown by headings. Returns list of (heading, content)."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            if current_lines or current_heading:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_heading, content))
            current_heading = match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_heading:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_heading, content))

    return sections


def _split_by_size(text: str, max_size: int, overlap: int) -> list[str]:
    """Split text by character count with sliding window overlap."""
    if len(text) <= max_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunks.append(text[start:end])
        start += max_size - overlap

    return chunks


def split_document(
    text: str, filename: str, max_size: int = 400, overlap: int = 50
) -> list[Chunk]:
    """Split document using hybrid strategy: headings first, then size."""
    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    position = 0

    if _has_headings(text):
        sections = _split_by_headings(text)
        for heading, content in sections:
            if len(content) <= max_size:
                chunks.append(
                    Chunk(
                        text=content,
                        metadata={
                            "filename": filename,
                            "heading": heading,
                            "position": position,
                        },
                    )
                )
                position += 1
            else:
                parts = _split_by_size(content, max_size, overlap)
                for part in parts:
                    chunks.append(
                        Chunk(
                            text=part,
                            metadata={
                                "filename": filename,
                                "heading": heading,
                                "position": position,
                            },
                        )
                    )
                    position += 1
    else:
        parts = _split_by_size(text, max_size, overlap)
        for part in parts:
            chunks.append(
                Chunk(
                    text=part,
                    metadata={
                        "filename": filename,
                        "heading": "",
                        "position": position,
                    },
                )
            )
            position += 1

    return chunks
