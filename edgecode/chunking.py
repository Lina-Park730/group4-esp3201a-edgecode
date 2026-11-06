"""Deterministic fixed-size and Python AST-aware chunking."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from .models import Chunk, JsonScalar, normalize_repo_path
from .repository import (
    discover_python_files,
    read_python_source,
    repository_relative_path,
)


def _validate_window(max_lines: int, overlap_lines: int) -> None:
    if max_lines < 1:
        raise ValueError("max_lines must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines cannot be negative")
    if overlap_lines >= max_lines:
        raise ValueError("overlap_lines must be smaller than max_lines")


def _source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def _chunks_for_span(
    lines: list[str],
    *,
    path: str,
    start_line: int,
    end_line: int,
    max_lines: int,
    overlap_lines: int,
    strategy: str,
    kind: str,
    symbol: str | None = None,
    breadcrumb: str | None = None,
    metadata: Mapping[str, JsonScalar] | None = None,
) -> list[Chunk]:
    """Chunk one inclusive source span without crossing its boundaries."""

    if start_line > end_line:
        return []
    _validate_window(max_lines, overlap_lines)
    step = max_lines - overlap_lines
    chunks: list[Chunk] = []
    cursor = start_line
    part = 1
    total_parts = max(1, (end_line - start_line - overlap_lines) // step + 1)
    while cursor <= end_line:
        part_end = min(cursor + max_lines - 1, end_line)
        part_metadata = dict(metadata or {})
        if total_parts > 1:
            part_metadata.update({"part": part, "parts": total_parts})
        chunks.append(
            Chunk(
                path=path,
                start_line=cursor,
                end_line=part_end,
                content="".join(lines[cursor - 1 : part_end]),
                strategy=strategy,
                kind=kind,
                symbol=symbol,
                breadcrumb=breadcrumb,
                metadata=part_metadata,
            )
        )
        if part_end == end_line:
            break
        cursor += step
        part += 1
    return chunks


def fixed_size_chunks(
    source: str,
    path: str,
    *,
    max_lines: int = 80,
    overlap_lines: int = 10,
    strategy: str = "fixed",
    metadata: Mapping[str, JsonScalar] | None = None,
) -> list[Chunk]:
    """Split source into overlapping, fixed-size line windows."""

    _validate_window(max_lines, overlap_lines)
    lines = _source_lines(source)
    if not lines:
        return []
    return _chunks_for_span(
        lines,
        path=normalize_repo_path(path),
        start_line=1,
        end_line=len(lines),
        max_lines=max_lines,
        overlap_lines=overlap_lines,
        strategy=strategy,
        kind="fixed",
        metadata=metadata,
    )


_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _decorated_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *(item.lineno for item in decorators)])  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class _SemanticSpan:
    start: int
    end: int
    kind: str
    symbol: str
    breadcrumb: str
    parent: str | None = None


def _semantic_spans(tree: ast.Module, path: str) -> list[_SemanticSpan]:
    spans: list[_SemanticSpan] = []
    for node in tree.body:
        if isinstance(node, _FUNCTION_NODES):
            spans.append(
                _SemanticSpan(
                    start=_decorated_start(node),
                    end=node.end_lineno or node.lineno,
                    kind=("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"),
                    symbol=node.name,
                    breadcrumb=f"{path}::{node.name}",
                )
            )
        elif isinstance(node, ast.ClassDef):
            methods = [item for item in node.body if isinstance(item, _FUNCTION_NODES)]
            if not methods:
                spans.append(
                    _SemanticSpan(
                        start=_decorated_start(node),
                        end=node.end_lineno or node.lineno,
                        kind="class",
                        symbol=node.name,
                        breadcrumb=f"{path}::{node.name}",
                    )
                )
                continue
            for method in methods:
                spans.append(
                    _SemanticSpan(
                        start=_decorated_start(method),
                        end=method.end_lineno or method.lineno,
                        kind=(
                            "async_method"
                            if isinstance(method, ast.AsyncFunctionDef)
                            else "method"
                        ),
                        symbol=method.name,
                        breadcrumb=f"{path}::{node.name}.{method.name}",
                        parent=node.name,
                    )
                )
    return sorted(spans, key=lambda item: (item.start, item.end, item.breadcrumb))


def _uncovered_spans(line_count: int, covered: Iterable[_SemanticSpan]) -> Iterator[tuple[int, int]]:
    mask = [False] * (line_count + 1)
    for span in covered:
        for line_number in range(max(1, span.start), min(line_count, span.end) + 1):
            mask[line_number] = True
    cursor = 1
    while cursor <= line_count:
        while cursor <= line_count and mask[cursor]:
            cursor += 1
        if cursor > line_count:
            break
        start = cursor
        while cursor <= line_count and not mask[cursor]:
            cursor += 1
        yield start, cursor - 1


def ast_aware_chunks(
    source: str,
    path: str,
    *,
    max_lines: int = 120,
    overlap_lines: int = 10,
) -> list[Chunk]:
    """Chunk Python around top-level functions and direct class methods.

    Semantic units remain whole when they fit within ``max_lines``.  Oversized
    units are split internally and retain the same symbol/breadcrumb metadata.
    Module and class context not belonging to a semantic unit is chunked into
    bounded windows.  Invalid Python safely falls back to fixed-size chunks.
    """

    _validate_window(max_lines, overlap_lines)
    normalized_path = normalize_repo_path(path)
    lines = _source_lines(source)
    if not lines:
        return []
    try:
        tree = ast.parse(source, filename=normalized_path)
    except (SyntaxError, ValueError) as error:
        message = getattr(error, "msg", str(error))
        return fixed_size_chunks(
            source,
            normalized_path,
            max_lines=max_lines,
            overlap_lines=overlap_lines,
            strategy="ast_fallback",
            metadata={
                "fallback": True,
                "parse_error": str(message),
                "parse_error_line": int(getattr(error, "lineno", 0) or 0),
            },
        )

    semantic = _semantic_spans(tree, normalized_path)
    output: list[Chunk] = []
    for span in semantic:
        span_metadata: dict[str, JsonScalar] = {"semantic_unit": True}
        if span.parent:
            span_metadata["parent"] = span.parent
        output.extend(
            _chunks_for_span(
                lines,
                path=normalized_path,
                start_line=span.start,
                end_line=span.end,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
                strategy="ast",
                kind=span.kind,
                symbol=span.symbol,
                breadcrumb=span.breadcrumb,
                metadata=span_metadata,
            )
        )
    for start, end in _uncovered_spans(len(lines), semantic):
        output.extend(
            _chunks_for_span(
                lines,
                path=normalized_path,
                start_line=start,
                end_line=end,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
                strategy="ast",
                kind="context",
                breadcrumb=f"{normalized_path}::<module>",
                metadata={"semantic_unit": False},
            )
        )
    return sorted(output, key=lambda chunk: (chunk.start_line, chunk.end_line, chunk.chunk_id))


def chunk_repository(
    repository_root: str | Path,
    *,
    strategy: str = "fixed",
    max_lines: int = 80,
    overlap_lines: int = 10,
    include_prefixes: Iterable[str] | None = None,
) -> list[Chunk]:
    """Discover and chunk selected Python files in a repository.

    ``include_prefixes`` uses normalized repository-relative paths. A directory
    prefix such as ``src/package/`` includes every Python file below it.
    """

    if strategy not in {"fixed", "ast"}:
        raise ValueError("strategy must be 'fixed' or 'ast'")
    root = Path(repository_root).expanduser().resolve()
    normalized_prefixes = tuple(
        normalize_repo_path(prefix).rstrip("/") for prefix in (include_prefixes or ())
    )
    chunks: list[Chunk] = []
    for file_path in discover_python_files(root):
        relative = repository_relative_path(file_path, root)
        if normalized_prefixes and not any(
            relative == prefix or relative.startswith(prefix + "/")
            for prefix in normalized_prefixes
        ):
            continue
        source = read_python_source(file_path)
        if strategy == "fixed":
            file_chunks = fixed_size_chunks(
                source,
                relative,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
            )
        else:
            file_chunks = ast_aware_chunks(
                source,
                relative,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
            )
        chunks.extend(file_chunks)
    return chunks


# Readable aliases for experiment configuration files.
chunk_fixed_size = fixed_size_chunks
chunk_ast_aware = ast_aware_chunks
