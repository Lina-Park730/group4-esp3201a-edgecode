"""Dense embedding adapter and cosine retrieval."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .models import Chunk, SearchResult


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _as_embedding_matrix(values: Iterable[Any], expected_rows: int) -> NDArray[np.float32]:
    rows = list(values)
    if expected_rows == 0:
        return np.empty((0, 0), dtype=np.float32)
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim == 1 and expected_rows == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != expected_rows:
        raise ValueError(
            f"Embedder returned shape {matrix.shape}; expected {expected_rows} rows"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values")
    return matrix


def _call_embedder(method: Any, texts: Sequence[str], batch_size: int) -> NDArray[np.float32]:
    try:
        values = method(list(texts), batch_size=batch_size)
    except TypeError:
        # Lightweight test doubles and some fastembed versions omit batch_size.
        values = method(list(texts))
    return _as_embedding_matrix(values, len(texts))


def normalize_embeddings(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix, dtype=np.float32),
        where=norms > 0,
    )


def cosine_retrieve(
    query_embedding: Sequence[float] | NDArray[np.floating[Any]],
    document_embeddings: NDArray[np.floating[Any]],
    chunks: Sequence[Chunk],
    *,
    k: int = 10,
    source: str = "dense",
) -> list[SearchResult]:
    """Rank chunks by cosine similarity with deterministic tie breaking."""

    if k < 0:
        raise ValueError("k cannot be negative")
    documents = np.asarray(document_embeddings, dtype=np.float32)
    if documents.ndim != 2:
        raise ValueError("document_embeddings must be a two-dimensional matrix")
    if documents.shape[0] != len(chunks):
        raise ValueError("The number of embeddings and chunks must match")
    if not chunks or k == 0:
        return []
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if documents.shape[1] != query.shape[0]:
        raise ValueError("Query and document embedding dimensions must match")
    document_norms = np.linalg.norm(documents, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominators = document_norms * query_norm
    dot_products = documents @ query
    similarities = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=np.float32),
        where=denominators > 0,
    )
    order = sorted(
        range(len(chunks)),
        key=lambda index: (-float(similarities[index]), chunks[index].chunk_id),
    )[:k]
    return [
        SearchResult(
            chunk=chunks[index],
            score=float(similarities[index]),
            rank=rank,
            source=source,
        )
        for rank, index in enumerate(order, start=1)
    ]


class DenseIndex:
    """In-memory dense index backed by fastembed's ``TextEmbedding``.

    Pass ``embedder=...`` to inject a compatible local/test model.  Import and
    construction of fastembed are deliberately lazy so importing this package
    never downloads a model.
    """

    def __init__(
        self,
        chunks: Iterable[Chunk],
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedder: Any | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.chunks = tuple(chunks)
        self.model_name = model_name
        self.batch_size = batch_size
        if embedder is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Dense retrieval requires fastembed; install project dependencies "
                    "or pass an embedder instance"
                ) from error
            embedder = TextEmbedding(model_name=model_name)
        self.embedder = embedder
        texts = [chunk.search_text for chunk in self.chunks]
        self.embeddings = _call_embedder(self.embedder.embed, texts, batch_size)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        method = getattr(self.embedder, "query_embed", self.embedder.embed)
        matrix = _call_embedder(method, [query], self.batch_size)
        if self.embeddings.shape[0] and matrix.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query and document embedder dimensions differ")
        return matrix[0]

    def search(self, query: str, *, k: int = 10) -> list[SearchResult]:
        if not self.chunks:
            return []
        return cosine_retrieve(
            self.embed_query(query), self.embeddings, self.chunks, k=k, source="dense"
        )


DenseRetriever = DenseIndex
