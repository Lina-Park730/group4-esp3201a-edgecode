#!/usr/bin/env python3
"""Retrieve Click/Python code and ask a local Ollama model about it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from edgecode.generation import OllamaClient, answer_question, render_answer
from edgecode.workflow import Workflow

ROOT = Path(__file__).resolve().parent




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config/pipeline.json")
    parser.add_argument("question", nargs="?", help="Question about the indexed code")
    parser.add_argument("--model", required=True, help="Exact model name from `ollama list`")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--check", action="store_true", help="Check Ollama/model without indexing")
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument(
        "--include-prefix", action="append",
        help="Limit indexing to a repository-relative directory; repeat to add directories",
    )
    parser.add_argument(
        "--arm", choices=("fixed_dense", "ast_dense", "ast_hybrid"),
        default="ast_hybrid", help="Retrieval setup to use (default: ast_hybrid)",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--context-chars", type=int, default=24000)
    parser.add_argument("--output", type=Path, help="Optional path to save answer and source metadata as JSON")
    args = parser.parse_args()

    if not args.check and (not args.question or not args.question.strip()):
        parser.error("Provide a question or use --check")
    if min(args.timeout, args.num_ctx, args.max_tokens, args.context_chars) <= 0:
        parser.error("Timeout, context size, token limit, and context budget must be positive")
    if args.num_ctx <= args.max_tokens:
        parser.error("--num-ctx must be greater than --max-tokens")

    try:
        client = OllamaClient(args.model, args.ollama_url, args.timeout)
        version = client.check()
        if args.check:
            print(f"Ollama {version}; model {args.model} is available.")
            return 0

        flow=Workflow(args.config)
        selected=flow.state(ready=True)
        RETRIEVAL=dict(flow.cfg, fixed_max_lines=selected["sizes"]["fixed"],
            ast_max_lines=selected["sizes"]["ast"], bm25_weight=selected["alpha"],
            dense_weight=1-selected["alpha"], k=selected["k"])
        if args.repo is None:args.repo=flow.repo

        from edgecode.bm25 import BM25Index
        from edgecode.chunking import chunk_repository
        from edgecode.dense import DenseIndex
        from edgecode.fusion import weighted_reciprocal_rank_fusion

        prefixes = args.include_prefix
        if prefixes is None and args.repo.resolve() == (ROOT / "repository/click-8.1.7").resolve():
            prefixes = ["src/click/"]

        is_fixed = args.arm == "fixed_dense"
        strategy = "fixed" if is_fixed else "ast"
        max_lines = RETRIEVAL["fixed_max_lines"] if is_fixed else RETRIEVAL["ast_max_lines"]
        chunks = chunk_repository(
            args.repo, strategy=strategy, max_lines=max_lines,
            overlap_lines=RETRIEVAL["overlap_lines"], include_prefixes=prefixes,
        )
        if not chunks:
            raise ValueError("No Python chunks found; check --repo and --include-prefix.")

        print(f"Indexing {len(chunks)} chunks ({args.arm})...", file=sys.stderr)
        dense_index = DenseIndex(
            chunks, model_name=RETRIEVAL["embedding_model"],
            batch_size=RETRIEVAL["embedding_batch_size"],
        )
        if args.arm == "ast_hybrid":
            dense_candidates = dense_index.search(args.question, k=RETRIEVAL["candidate_k"])
            bm25_candidates = BM25Index(chunks).search(args.question, k=RETRIEVAL["candidate_k"])
            results = weighted_reciprocal_rank_fusion(
                {"bm25": bm25_candidates, "dense": dense_candidates},
                weights={"bm25": RETRIEVAL["bm25_weight"], "dense": RETRIEVAL["dense_weight"]},
                rrf_k=RETRIEVAL["rrf_k"], limit=RETRIEVAL["k"],
            )
        else:
            results = dense_index.search(args.question, k=RETRIEVAL["k"])

        answer = answer_question(
            client, args.question, results, max_chars=args.context_chars,
            num_ctx=args.num_ctx, max_tokens=args.max_tokens,
        )
        answer.update(
            question=args.question, model=args.model, ollama_version=version,
            arm=args.arm, retrieval_parameters=RETRIEVAL,
        )
        print(render_answer(answer))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(answer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
