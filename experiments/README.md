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
