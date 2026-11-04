"""Measure the unchanged notebook baselines on the frozen question selection."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import ValidationError


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "1_Long_Memory.ipynb"
REPORT = ROOT / "experiments" / "graph_rag_audit_20261104" / "report"
METHODS = {
    "no_memory": "chat_no_memory",
    "chat_history": "chat_with_history",
    "summarization": "chat_with_summary",
    "history_rag": "chat_with_history_rag",
}


class ModelTimer(BaseCallbackHandler):
    def __init__(self):
        self.started = {}
        self.calls = []

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.started[run_id] = perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        metadata = response.generations[0][0].message.response_metadata
        self.calls.append({
            "seconds": perf_counter() - self.started.pop(run_id),
            **{key: metadata.get(key) for key in (
                "model", "load_duration", "prompt_eval_duration", "eval_duration",
                "prompt_eval_count", "eval_count", "done_reason",
            )},
        })


def load_cells(namespace, methods):
    notebook = json.loads(NOTEBOOK.read_bytes())
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    targets = ["chat", "path", "dated_sessions", *(METHODS[m] for m in methods), "grade_answer"]
    selected = [cells[0]]
    for target in targets:
        matches = []
        for source in cells:
            names = set()
            for node in ast.parse(source).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            if target in names:
                matches.append(source)
        if len(matches) != 1:
            raise RuntimeError(f"Expected one cell defining {target}.")
        selected.append(matches[0])
    for number, source in enumerate(selected):
        exec(compile(source, f"{NOTEBOOK.name}:baseline_cell_{number}", "exec"), namespace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--split", choices=("development", "validation", "all", "regression"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = json.loads((REPORT / "sample_manifest.json").read_text())
    indices = manifest["development"] + manifest["validation"] if args.split == "all" else manifest[args.split]
    if args.limit:
        indices = indices[:args.limit]
    digest = hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
    completed = set()
    if args.output.exists():
        previous = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
        if previous[0]["notebook_sha256"] != digest:
            raise RuntimeError("Notebook changed since this run started.")
        completed = {(row["method"], row["dataset_index"]) for row in previous if row["record_type"] == "prediction"}
    start = perf_counter()
    ns = {"__name__": "baseline_comparison", "__file__": str(NOTEBOOK)}
    load_cells(ns, args.methods)
    if ns["seed"] != manifest["seed"] or not ns["JUDGE_OK"]:
        raise RuntimeError("Seed mismatch or failed judge controls.")
    data_digest = hashlib.sha256(Path(ns["path"]).read_bytes()).hexdigest()
    frozen = json.loads((REPORT / "summary.json").read_text())
    if data_digest != frozen["data"]["sha256"]:
        raise RuntimeError("Dataset hash differs from the frozen comparison.")
    setup_seconds = perf_counter() - start
    answer_timer, summary_timer = ModelTimer(), ModelTimer()
    ns["chat"].callbacks = [answer_timer]
    if "summarizer" in ns:
        ns["summarizer"].model.callbacks = [summary_timer]
    index_times = []
    if "history_index" in ns:
        original_index = ns["history_index"]

        def timed_index(turns):
            indexed = perf_counter()
            result = original_index(turns)
            index_times.append(perf_counter() - indexed)
            return result

        ns["history_index"] = timed_index
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        def save(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

        if not completed:
            save({
                "record_type": "run", "started_utc": datetime.now(timezone.utc).isoformat(),
                "notebook_sha256": digest, "dataset_sha256": data_digest,
                "dataset_indices": indices, "methods": args.methods,
                "setup_seconds": setup_seconds, "judge_controls": "6/6",
                "timing_note": "One sample per call; wall time includes cold memory construction and any in-call model load. Setup and judge controls precede timing, warming the answer model. Grading is timed separately. No graph rebuild.",
            })
        for method in args.methods:
            for index in indices:
                if (method, index) in completed:
                    continue
                sample = ns["dataset"][index]
                answer_timer.calls.clear()
                summary_timer.calls.clear()
                index_times.clear()
                print(f"START {method} {index} {sample['question_id']}", flush=True)
                start = perf_counter()
                result = ns[METHODS[method]]([sample])[0]
                elapsed = perf_counter() - start
                answer = result["answer"]
                message = answer["messages"][-1] if isinstance(answer, dict) else answer
                case = {
                    "type": "abstention" if sample["question_id"].endswith("_abs") else sample["question_type"],
                    "question": sample["question"], "reference": sample["answer"], "candidate": message.content,
                }
                grading = perf_counter()
                grading_error = None
                try:
                    automatic = ns["grade_answer"](case).correct
                except ValidationError as error:
                    automatic = None
                    grading_error = f"{type(error).__name__}: {error}"
                row = {
                    "record_type": "prediction", "method": method, "dataset_index": index,
                    "id": sample["question_id"], **case, "automatic_correct": automatic,
                    "seconds": elapsed, "grading_seconds": perf_counter() - grading,
                    "grading_error": grading_error,
                    "answer_calls": answer_timer.calls.copy(), "summary_calls": summary_timer.calls.copy(),
                    "index_seconds": sum(index_times) if index_times else None,
                    "summary": result.get("summary"), "messages": result.get("messages"),
                    "response_metadata": message.response_metadata,
                }
                save(row)
                print(f"SAVED {method} {index}: correct={automatic}, seconds={elapsed:.2f}, summary_calls={len(summary_timer.calls)}", flush=True)


if __name__ == "__main__":
    main()
