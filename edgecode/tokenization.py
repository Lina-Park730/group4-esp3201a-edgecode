"""Identifier-aware lexical tokenization."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_CAMEL_BOUNDARY_1 = re.compile(r"([a-z0-9])([A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def split_identifier(identifier: str) -> list[str]:
    """Split snake_case and camelCase while keeping acronym groups intact."""

    pieces: list[str] = []
    for snake_part in identifier.strip("_").split("_"):
        if not snake_part:
            continue
        separated = _CAMEL_BOUNDARY_2.sub(r"\1 \2", snake_part)
        separated = _CAMEL_BOUNDARY_1.sub(r"\1 \2", separated)
        pieces.extend(part.lower() for part in separated.split() if part)
    return pieces


def lexical_tokenize(text: str) -> list[str]:
    """Tokenize code/text, retaining full identifiers and their components.

    For example, ``getHTTP_response`` becomes ``gethttp_response``, ``get``,
    ``http``, and ``response``.  Keeping both forms supports exact symbol lookup
    without sacrificing natural-language matching.
    """

    output: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        normalized = token.lower()
        output.append(normalized)
        if token[0].isalpha() or token[0] == "_":
            for component in split_identifier(token):
                if component != normalized:
                    output.append(component)
    return output


# Short alias convenient for experiment configuration.
tokenize_code = lexical_tokenize
