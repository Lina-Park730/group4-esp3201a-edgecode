# EdgeCode Assistant

Code retrieval and question answering over the Click 8.1.7 source code. The project compares fixed-window chunking with Python AST chunking, using dense retrieval and a BM25–dense hybrid.

| Method | Chunking | Retrieval |
| --- | --- | --- |
| Fixed Dense | Overlapping line windows | Dense embeddings |
| AST Dense | Functions, methods and context windows | Dense embeddings |
| AST Hybrid | Same chunks as AST Dense | BM25 and dense retrieval with weighted reciprocal rank fusion |

## Setup

Requires Python 3.10 or later. Ollama is optional and is used only for answer generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Download the [Click 8.1.7 source archive](https://github.com/pallets/click/archive/refs/tags/8.1.7.zip), extract it, and place the `click-8.1.7` folder inside `repository/`. The source path should be:

```text
repository/click-8.1.7/src/click/
```

The Click source is an external experimental corpus and is not included in this repository. Installing Click with pip does not create the required source folder. FastEmbed downloads the embedding model on first use.

## Run the experiments

Run commands from the project root.

### 1. Development

```bash
python3 run_pipeline.py develop
```

The 20 development questions are used in three stages:

1. **Chunk size:** compare Fixed Dense and AST Dense at 60, 70, 80 and 90 maximum lines. Select one shared size by the highest mean MRR@5 across both methods. Ties prefer the smaller MRR gap, then the smaller size.
2. **Hybrid weight:** compare BM25 weights from 0.0 to 1.0 in steps of 0.1 at the selected AST size. Select the highest MRR@5; ties prefer the smaller BM25 weight.
3. **Top-K:** report results for K = 3, 5 and 7 across all three methods. K is selected manually before evaluation.

The stages can also run separately, in order:

```bash
python3 run_pipeline.py sizes
python3 run_pipeline.py alpha
python3 run_pipeline.py k
```

### 2. Retrieval evaluation

Review the development comparison in `outputs/development/k_*/comparison.md`, then pass the chosen K. For example, to use K = 5:

```bash
python3 run_pipeline.py evaluate --k 5
```

This freezes the selected parameters and evaluates all three methods on 80 questions. Subsequent evaluation runs use the frozen K:

```bash
python3 run_pipeline.py evaluate
```

Results include overall and per-category metrics, per-question rankings, gold-span diagnostics, indexing times and retrieval latency. The five question categories cover identifiers, semantic behaviour, cross-file retrieval, error cases and code explanations.

Selections are stored in `outputs/development/selected_parameters.json`. Configuration, source or dataset changes invalidate existing selections. Rerun development after making those changes.

### 3. Answer generation with Ollama

Start Ollama and install a model before running generation. Set `llm.model` in `config/pipeline.json` to its installed name before development, or pass the model name with `--model`.

```bash
ollama list
python3 run_pipeline.py llm-check --model MODEL_NAME
python3 run_pipeline.py llm --model MODEL_NAME --limit 20 --fail-fast
```

Replace `MODEL_NAME` with a name shown by `ollama list`. The supplied configuration uses `gemma4:e4b`; the model must be available locally to use that setting.

The 20-question run generates 60 answers: one answer per question for each method. Questions are selected deterministically across categories. Use `--limit 5` for a smaller check or `--limit 80` for the full set.

Each run saves answers, supplied context, citations, timing, raw responses and a `manual_review.csv` for scoring correctness, completeness, citation support and cross-function reasoning. Citation checks validate source IDs and line ranges; answer quality requires reviewing the content.

### Interactive questions

After development and retrieval evaluation:

```bash
python3 ask.py --model MODEL_NAME --check
python3 ask.py "How does Context.invoke handle defaults?" --model MODEL_NAME --arm ast_hybrid
```

Use `--help` on an entry point to view its options.

## Configuration

Experiment settings are in `config/pipeline.json`.

| Setting | Value |
| --- | --- |
| Corpus | Click 8.1.7, `src/click/` |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Maximum chunk lines | 60, 70, 80, 90 |
| Window overlap | 10 lines |
| Candidates per retrieval branch | 20 |
| RRF constant | 60 |
| Development Top-K values | 3, 5, 7 |
| Gold-span coverage threshold | 25%, with at least one overlapping line |
| Retrieval timing | 1 warm-up and 3 measured repetitions per query |
| Generation context budget | 3,000 using the configured token counter |

AST chunking preserves functions and methods that fit within the line limit. Oversized units are split, and uncovered module or class context is also chunked.

For hybrid retrieval, alpha is the BM25 weight:

```text
score = alpha / (60 + BM25 rank) + (1 - alpha) / (60 + dense rank)
```

Missing branches contribute zero. Alpha = 0 uses dense retrieval only; alpha = 1 uses BM25 only. Interior weights combine up to 40 unique candidates.

The default context counter uses UTF-8 bytes as a token estimate, not exact model tokenization. Set `llm.tokenizer_path` to a matching local `tokenizer.json` to use tokenizer-based counts. Whole chunks are included in rank order when they fit the budget; oversized chunks are skipped.

## Metrics

- **MRR@K:** reciprocal rank of the first matching chunk, averaged over questions.
- **Recall@K:** fraction of distinct gold spans matched within the retrieved chunks, averaged over questions.
- **Precision@K:** matching chunks divided by the number of returned chunks, averaged over questions.
- **Gold-hit questions:** questions with at least one gold-span match in the top K.

A match requires the same file path and sufficient line overlap. Matching several chunks to the same span does not increase recall. Evaluation reports both the selected K and reference metrics at K = 5.

The evaluation set was examined in earlier experiments. It is not a newly untouched test set. Duplicate checks catch exact overlaps, but do not establish semantic independence between the development and evaluation questions.

## Project files

| Path | Contents |
| --- | --- |
| `edgecode/` | Chunking, retrieval, evaluation, generation and experiment workflow |
| `config/` | Experiment configuration |
| `data/` | Development, evaluation and original reference questions |
| `tests/` | Retrieval workflow and generation tests |
| `run_pipeline.py` | Main experiment CLI |
| `ask.py` | Interactive question answering |
| `review_gold_spans.py` | Export source evidence for annotation review |
| `run_*.py` | Short entry points for individual experiment stages |
| `repository/` | External Click source; excluded from Git |
| `outputs/` | Generated selections, metrics, plots and answers; excluded from Git |

## Checks

```bash
python3 -m unittest discover -s tests -v
python3 run_pipeline.py --help
python3 ask.py --help
```

The tests use synthetic embeddings and mocked Ollama responses. Live retrieval and generation require the embedding model, Click source and a running Ollama server for the generation stage.
