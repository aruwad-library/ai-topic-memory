# Reproducing the Memory Experiments

The notebook is the source of the current implementation. `reproduce.py` loads its setup, models, dataset, helper functions, RAG methods, and judge controls without copying or modifying their logic. It requires the notebook's Python dependencies, a running local Ollama server with `qwen3:8b` and `qwen3:4b-instruct`, and `SEED=42` in the project `.env`.

## Refresh the Saved Notebook

Run every cell in a fresh kernel, equivalent to Restart Kernel & Run All:

```powershell
.\.venv\Scripts\python.exe experiments\execute_notebook.py
```

This uses the invoking Python environment and saves `1_Long_Memory.ipynb` only after all cells complete without errors. Cell sources remain unchanged. It runs the existing baseline examples, judge controls, and two-question Graph RAG evaluation, not the 28-question comparison. The execution record is saved as `graph_rag_audit_20261104/report/notebook_execution.json`; checkpoints or failed runs remain under `tmp/notebook_execution/`. The execution helper requires `nbclient`, included in the project requirements.

## Rerun Selected RAG Examples

From the project root, preview the selected examples without loading models:

```powershell
.\.venv\Scripts\python.exe experiments\reproduce.py --split all
```

Run the same two Graph RAG regression examples used by the notebook's evaluation cell:

```powershell
.\.venv\Scripts\python.exe experiments\reproduce.py --split regression --method graph --run
```

Rerun both current RAG methods on the frozen 28-example selection only when a longer experiment is intended:

```powershell
.\.venv\Scripts\python.exe experiments\reproduce.py --split all --run
```

Use `--split development` or `--split validation` for 14 examples, and `--limit 2` for a smaller subset. New predictions, evidence, references, timings, and automatic grades are written to a new JSONL file under `tmp/`; `--output` can select another new file. The runner refuses to overwrite existing files or write into the frozen report folder. It checks the dataset SHA-256 and stops if judge controls fail.

## Retained Evidence

`graph_rag_audit_20261104/report/` contains the final comparison, sample manifest, failure diagnosis, dataset audit, and validation records. Start with `comparison.csv` and `summary.json`. These preserve the earlier 28-question comparison and its two independent, method-blind model reviews.

The runner regenerates predictions with the current notebook implementation; it does not regenerate those independent reviews or promise identical scores. The local judge can make semantic mistakes even after passing controls. Model/runtime versions, graph extraction, and generation may affect reruns. Cached/concurrent historical timings are not comparable to a fresh run's latency.

The notebook's saved two-question execution is a runnable demonstration. It is separate from the frozen 28-question evidence and does not replace it. Disposable caches, exploratory variants, duplicate snapshots, and raw debugging logs are not required by this runner.

## Supplemental Baselines and Runtime

`baseline_extension_20261105/` adds No Memory, Chat History and Summarization on the same 28 questions. `predictions.csv` retains answers, automatic grades, two independent reviews with method labels masked, rationales and timing. `review_summary.json` reports No Memory 2-3/28 (one ambiguous location answer), Chat History 4/28 and Summarization 6/28. The frozen RAG scores remain unchanged. These are model-assisted reviews, not official GPT-4o evaluation or human ground truth.

`runtime.csv` measures four methods sequentially on the same two regression questions; `metadata.json` records configuration, hashes, hardware, timing boundaries and an interrupted judge call. Setup, controls and grading are excluded from per-case timings. Memory is rebuilt or replayed per case. The saved Graph times come from a separate earlier run, so they do not support a controlled speed ratio.

```powershell
.\.venv\Scripts\python.exe experiments\compare_baselines.py --methods no_memory chat_history --output tmp\new_fast.jsonl
.\.venv\Scripts\python.exe experiments\compare_baselines.py --methods summarization --output tmp\new_summary.jsonl
.\.venv\Scripts\python.exe experiments\compare_baselines.py --methods no_memory chat_history summarization history_rag --split regression --output tmp\new_runtime.jsonl
```

The runner imports the unchanged notebook definitions and logs grading parse failures as ungraded. Automatic labels are not reliable enough to replace independent review. Summarization retains the notebook's unspecified temperature/seed defaults; new generations may differ. Structured Memory is excluded because it extracts records per message. Raw JSONL and review packets remain local and ignored; the compact CSVs preserve the final evidence.
