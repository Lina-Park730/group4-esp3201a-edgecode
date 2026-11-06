"""Small, dependency-free BM25 implementation for code chunks."""

from __future__ import annotations

from collections import Counter
import math
from typing import Callable, Iterable

from .models import Chunk, SearchResult
from .tokenization import lexical_tokenize


Tokenizer = Callable[[str], list[str]]


class BM25Index:
    """An immutable in-memory Okapi BM25 index."""

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        tokenizer: Tokenizer = lexical_tokenize,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        self.chunks = tuple(chunks)
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.term_frequencies = tuple(
            Counter(tokenizer(chunk.search_text)) for chunk in self.chunks
        )
        self.document_lengths = tuple(
            sum(frequencies.values()) for frequencies in self.term_frequencies
        )
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        document_count = len(self.chunks)
        self.inverse_document_frequency = {
            term: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        if not self.chunks:
            return []
        query_terms = self.tokenizer(query)
        scores = [0.0] * len(self.chunks)
        average_length = self.average_document_length or 1.0
        for index, frequencies in enumerate(self.term_frequencies):
            length_normalizer = 1.0 - self.b + self.b * (
                self.document_lengths[index] / average_length
            )
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * length_normalizer
                score += self.inverse_document_frequency.get(term, 0.0) * numerator / denominator
            scores[index] = score
        return scores

    def search(self, query: str, *, k: int = 10) -> list[SearchResult]:
        if k < 0:
            raise ValueError("k cannot be negative")
        scored = zip(self.chunks, self.scores(query))
        ordered = sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))[:k]
        return [
            SearchResult(chunk=chunk, score=score, rank=rank, source="bm25")
            for rank, (chunk, score) in enumerate(ordered, start=1)
        ]


BM25Retriever = BM25Index
