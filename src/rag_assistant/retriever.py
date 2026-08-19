from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9']+")
SUPPORTED_SUFFIXES = {".txt", ".md"}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "my", "of", "on", "or", "the",
    "to", "what", "when", "where", "which", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOP_WORDS
    ]


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    source: str
    text: str
    score: float


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Knowledge base requires at least one non-empty chunk.")
        self.chunks = chunks
        self._vectors, self._idf = self._build_index(chunks)

    @classmethod
    def from_directory(cls, directory: str | Path, chunk_size: int = 700) -> "KnowledgeBase":
        root = Path(directory)
        if not root.is_dir():
            raise ValueError(f"Document directory does not exist: {root}")

        chunks: list[Chunk] = []
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                text = path.read_text(encoding="utf-8").strip()
                for part in cls._chunk_text(text, chunk_size):
                    chunks.append(Chunk(source=path.name, text=part))
        return cls(chunks)

    @staticmethod
    def _chunk_text(text: str, chunk_size: int) -> list[str]:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters.")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _build_index(chunks: list[Chunk]) -> tuple[list[dict[str, float]], dict[str, float]]:
        token_counts = [Counter(tokenize(chunk.text)) for chunk in chunks]
        document_frequency: Counter[str] = Counter()
        for counts in token_counts:
            document_frequency.update(counts.keys())

        total = len(chunks)
        idf = {term: math.log((1 + total) / (1 + freq)) + 1 for term, freq in document_frequency.items()}
        return [KnowledgeBase._normalize({term: count * idf[term] for term, count in counts.items()}) for counts in token_counts], idf

    @staticmethod
    def _normalize(vector: dict[str, float]) -> dict[str, float]:
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {term: value / norm for term, value in vector.items()} if norm else {}

    def search(self, query: str, limit: int = 3, minimum_score: float = 0.08) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")
        counts = Counter(tokenize(query))
        query_vector = self._normalize({term: count * self._idf.get(term, 0.0) for term, count in counts.items()})

        ranked: list[SearchResult] = []
        for chunk, vector in zip(self.chunks, self._vectors):
            score = sum(query_vector.get(term, 0.0) * value for term, value in vector.items())
            if score >= minimum_score:
                ranked.append(SearchResult(chunk.source, chunk.text, round(score, 4)))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]
