"""Execute every notebook cell in a fresh kernel and save only a successful run."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "1_Long_Memory.ipynb"
REPORT = ROOT / "experiments/graph_rag_audit_20261104/report/notebook_execution.json"
WORK = ROOT / "tmp/notebook_execution"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    original_bytes = NOTEBOOK.read_bytes()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    sources = [(cell.cell_type, cell.source) for cell in notebook.cells]
    started = perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    cell_started = {}

    def start_cell(cell, cell_index, **kwargs):
        cell_started[cell_index] = perf_counter()
        print(f"START cell {cell_index}: {cell.source.splitlines()[0][:90]}", flush=True)

    def finish_cell(cell, cell_index, **kwargs):
        elapsed = perf_counter() - cell_started[cell_index]
        print(f"DONE cell {cell_index}: {elapsed:.1f}s", flush=True)
        nbformat.write(notebook, WORK / "checkpoint.ipynb")

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=3600, startup_timeout=120,
        allow_errors=False, resources={"metadata": {"path": str(ROOT)}},
        on_cell_execute=start_cell, on_cell_executed=finish_cell,
    )
    try:
        client.execute(cleanup_kc=True)
    except Exception:
        nbformat.write(notebook, WORK / "failed.ipynb")
        raise
    assert sources == [(cell.cell_type, cell.source) for cell in notebook.cells]
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    nbformat.validate(notebook)
    assert NOTEBOOK.read_bytes() == original_bytes, "Notebook changed during execution."
    saved = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    NOTEBOOK.write_text(saved, encoding="utf-8", newline="")
    report = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "fresh_kernel": True,
        "executed_code_cells": len(code_cells),
        "execution_counts": [cell.execution_count for cell in code_cells],
        "errors": 0,
        "all_cell_sources_unchanged": True,
        "before_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "saved_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "scope": "All cells, including the existing two-question Graph RAG evaluation; not the archived 28-question comparison.",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
