"""Repository discovery and source loading helpers."""

from __future__ import annotations

from pathlib import Path
import tokenize
from typing import Iterable


DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "venv",
    }
)


def discover_python_files(
    repository_root: str | Path,
    *,
    exclude_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    include_hidden: bool = False,
) -> list[Path]:
    """Discover Python files beneath ``repository_root`` in stable path order.

    Symlinked files that resolve outside the repository are ignored.  Returned
    paths are absolute, which makes subsequent reads independent of the process
    working directory.
    """

    root = Path(repository_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {root}")
    excluded = set(exclude_dirs)
    found: list[Path] = []
    for candidate in root.rglob("*.py"):
        relative = candidate.relative_to(root)
        parent_parts = relative.parts[:-1]
        if any(part in excluded for part in parent_parts):
            continue
        if not include_hidden and any(part.startswith(".") for part in relative.parts):
            continue
        if not candidate.is_file():
            continue
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        found.append(resolved)
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def repository_relative_path(path: str | Path, repository_root: str | Path) -> str:
    root = Path(repository_root).expanduser().resolve()
    return Path(path).expanduser().resolve().relative_to(root).as_posix()


def read_python_source(path: str | Path) -> str:
    """Read Python source while respecting a PEP 263 encoding declaration."""

    with tokenize.open(Path(path)) as handle:
        return handle.read()
