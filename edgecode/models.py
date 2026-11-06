"""Shared, deterministic data models for EdgeCode retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Mapping


JsonScalar = str | int | float | bool | None


def normalize_repo_path(path: str) -> str:
    """Return a stable repository-relative path using POSIX separators."""

    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return str(PurePosixPath(normalized))


def _stable_metadata(metadata: Mapping[str, JsonScalar]) -> dict[str, JsonScalar]:
    return {str(key): metadata[key] for key in sorted(metadata)}


@dataclass(frozen=True, slots=True)
class Chunk:
    """A line-addressable source-code chunk.

    ``start_line`` and ``end_line`` are inclusive and one-based.  The chunk ID is
    content-addressed, so identical inputs produce byte-for-byte identical JSON.
    """

    path: str
    start_line: int
    end_line: int
    content: str
    strategy: str
    kind: str = "code"
    symbol: str | None = None
    breadcrumb: str | None = None
    metadata: Mapping[str, JsonScalar] = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError("start_line must be at least 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line must not be before start_line")
        object.__setattr__(self, "path", normalize_repo_path(self.path))
        object.__setattr__(self, "metadata", _stable_metadata(self.metadata))
        if not self.chunk_id:
            identity = {
                "breadcrumb": self.breadcrumb,
                "content": self.content,
                "end_line": self.end_line,
                "kind": self.kind,
                "metadata": self.metadata,
                "path": self.path,
                "start_line": self.start_line,
                "strategy": self.strategy,
                "symbol": self.symbol,
            }
            payload = json.dumps(
                identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            object.__setattr__(self, "chunk_id", hashlib.sha256(payload).hexdigest()[:20])

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1

    @property
    def search_text(self) -> str:
        """Text supplied to lexical and embedding indexes, including provenance."""

        labels = [self.path]
        if self.breadcrumb:
            labels.append(self.breadcrumb)
        elif self.symbol:
            labels.append(self.symbol)
        return "\n".join(labels + [self.content])

    def to_dict(self) -> dict[str, Any]:
        """Serialize using a fixed key order and stable metadata order."""

        return {
            "chunk_id": self.chunk_id,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "strategy": self.strategy,
            "kind": self.kind,
            "symbol": self.symbol,
            "breadcrumb": self.breadcrumb,
            "content": self.content,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Chunk":
        return cls(
            chunk_id=str(value.get("chunk_id", "")),
            path=str(value["path"]),
            start_line=int(value["start_line"]),
            end_line=int(value["end_line"]),
            content=str(value["content"]),
            strategy=str(value["strategy"]),
            kind=str(value.get("kind", "code")),
            symbol=(None if value.get("symbol") is None else str(value["symbol"])),
            breadcrumb=(
                None if value.get("breadcrumb") is None else str(value["breadcrumb"])
            ),
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, value: str) -> "Chunk":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ranked retrieval result."""

    chunk: Chunk
    score: float
    rank: int
    source: str

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    # These provenance proxies make ranked results directly consumable by
    # prompt/context formatters without discarding their score and rank.
    @property
    def path(self) -> str:
        return self.chunk.path

    @property
    def start_line(self) -> int:
        return self.chunk.start_line

    @property
    def end_line(self) -> int:
        return self.chunk.end_line

    @property
    def content(self) -> str:
        return self.chunk.content

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": float(self.score),
            "source": self.source,
            "chunk": self.chunk.to_dict(),
        }
