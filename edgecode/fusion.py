"""Rank fusion for hybrid lexical and dense retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import Chunk, SearchResult


Rankings = Mapping[str, Sequence[SearchResult]] | Sequence[Sequence[SearchResult]]
Weights = Mapping[str, float] | Sequence[float]


def weighted_reciprocal_rank_fusion(
    rankings: Rankings,
    *,
    weights: Weights | None = None,
    rrf_k: int = 60,
    limit: int | None = None,
) -> list[SearchResult]:
    """Fuse rankings using ``sum(weight / (rrf_k + rank))``.

    A mapping gives each ranking an explicit name.  A sequence is named by its
    zero-based position.  Scores from the input systems do not need to be on the
    same scale, which is the main reason RRF is used for this baseline.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative")
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    if isinstance(rankings, Mapping):
        named_rankings = [(str(name), values) for name, values in rankings.items()]
    else:
        named_rankings = [(str(index), values) for index, values in enumerate(rankings)]

    if weights is None:
        named_weights = {name: 1.0 for name, _ in named_rankings}
    elif isinstance(weights, Mapping):
        named_weights = {name: float(weights.get(name, 1.0)) for name, _ in named_rankings}
    else:
        if len(weights) != len(named_rankings):
            raise ValueError("weights must match the number of rankings")
        named_weights = {
            name: float(weight) for (name, _), weight in zip(named_rankings, weights)
        }
    if any(weight < 0 for weight in named_weights.values()):
        raise ValueError("weights cannot be negative")

    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    best_rank: dict[str, int] = {}
    for name, ranking in named_rankings:
        if named_weights[name] == 0:
            continue
        seen_in_ranking: set[str] = set()
        for position, result in enumerate(ranking, start=1):
            chunk_id = result.chunk_id
            if chunk_id in seen_in_ranking:
                continue
            seen_in_ranking.add(chunk_id)
            chunks.setdefault(chunk_id, result.chunk)
            best_rank[chunk_id] = min(position, best_rank.get(chunk_id, position))
            scores[chunk_id] = scores.get(chunk_id, 0.0) + named_weights[name] / (
                rrf_k + position
            )

    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id),
    )
    if limit is not None:
        ordered_ids = ordered_ids[:limit]
    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=scores[chunk_id],
            rank=rank,
            source="weighted_rrf",
        )
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    ]


weighted_rrf = weighted_reciprocal_rank_fusion
