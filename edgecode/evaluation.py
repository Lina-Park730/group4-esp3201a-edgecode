"""Gold-span matching and retrieval metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .models import Chunk, SearchResult, normalize_repo_path


@dataclass(frozen=True, slots=True)
class GoldSpan:
    """An inclusive, one-based relevant source span."""

    path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError("start_line must be at least 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not be before start_line")
        object.__setattr__(self, "path", normalize_repo_path(self.path))

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoldSpan":
        return cls(
            path=str(value.get("path", value.get("file", ""))),
            start_line=int(value["start_line"]),
            end_line=int(value["end_line"]),
        )


def line_overlap(
    first_start: int, first_end: int, second_start: int, second_end: int
) -> int:
    """Return the number of shared lines between two inclusive spans."""

    return max(0, min(first_end, second_end) - max(first_start, second_start) + 1)


def chunk_span_overlap_lines(chunk: Chunk, gold: GoldSpan) -> int:
    if normalize_repo_path(chunk.path) != gold.path:
        return 0
    return line_overlap(chunk.start_line, chunk.end_line, gold.start_line, gold.end_line)


def chunk_overlaps_gold(
    chunk: Chunk,
    gold: GoldSpan,
    *,
    min_overlap_lines: int = 1,
    min_gold_coverage: float = 0.0,
) -> bool:
    """Whether a chunk sufficiently overlaps a gold source span."""

    if min_overlap_lines < 1:
        raise ValueError("min_overlap_lines must be at least 1")
    if not 0.0 <= min_gold_coverage <= 1.0:
        raise ValueError("min_gold_coverage must be between 0 and 1")
    overlap = chunk_span_overlap_lines(chunk, gold)
    return overlap >= min_overlap_lines and overlap / gold.line_count >= min_gold_coverage


def gold_coverage(chunk: Chunk, gold: GoldSpan) -> float:
    return chunk_span_overlap_lines(chunk, gold) / gold.line_count


def span_iou(chunk: Chunk, gold: GoldSpan) -> float:
    overlap = chunk_span_overlap_lines(chunk, gold)
    if not overlap:
        return 0.0
    union = chunk.line_count + gold.line_count - overlap
    return overlap / union


def _as_chunk(value: Chunk | SearchResult) -> Chunk:
    return value.chunk if isinstance(value, SearchResult) else value


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    k: int
    hit: float
    precision: float
    recall: float
    reciprocal_rank: float
    matching_chunks: int
    matched_gold_spans: int
    total_gold_spans: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "k": self.k,
            "hit_at_k": self.hit,
            "precision_at_k": self.precision,
            "recall_at_k": self.recall,
            "mrr_at_k": self.reciprocal_rank,
            "matching_chunks": self.matching_chunks,
            "matched_gold_spans": self.matched_gold_spans,
            "total_gold_spans": self.total_gold_spans,
        }


def retrieval_metrics_at_k(
    ranking: Sequence[Chunk | SearchResult],
    gold_spans: Iterable[GoldSpan],
    *,
    k: int = 10,
    min_overlap_lines: int = 1,
    min_gold_coverage: float = 0.0,
) -> RetrievalMetrics:
    """Compute hit, gold-span recall, and reciprocal rank at ``k``."""

    if k < 1:
        raise ValueError("k must be at least 1")
    gold = tuple(gold_spans)
    top_chunks = tuple(_as_chunk(item) for item in ranking[:k])
    matched_indices: set[int] = set()
    matching_chunks = 0
    first_matching_rank = 0
    for rank, chunk in enumerate(top_chunks, start=1):
        matches = {
            index
            for index, span in enumerate(gold)
            if chunk_overlaps_gold(
                chunk,
                span,
                min_overlap_lines=min_overlap_lines,
                min_gold_coverage=min_gold_coverage,
            )
        }
        if matches:
            matching_chunks += 1
            matched_indices.update(matches)
            if not first_matching_rank:
                first_matching_rank = rank
    total = len(gold)
    return RetrievalMetrics(
        k=k,
        hit=float(bool(matched_indices)),
        precision=(matching_chunks / len(top_chunks) if top_chunks else 0.0),
        recall=(len(matched_indices) / total if total else 0.0),
        reciprocal_rank=(1.0 / first_matching_rank if first_matching_rank else 0.0),
        matching_chunks=matching_chunks,
        matched_gold_spans=len(matched_indices),
        total_gold_spans=total,
    )


evaluate_retrieval = retrieval_metrics_at_k
