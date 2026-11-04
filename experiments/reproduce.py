"""Rerun the notebook's current RAG methods; preview the selection by default."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "experiments" / "graph_rag_audit_20261104" / "report"
NOTEBOOK = ROOT / "1_Long_Memory.ipynb"


def selected_cells(notebook):
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    targets = (
        "chat", "path", "dated_sessions", "chat_with_history_rag",
        "chat_with_graph_rag", "grade_answer",
    )
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
            raise RuntimeError(f"Expected one notebook cell defining {target!r}.")
        selected.append(matches[0])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "validation", "all", "regression"), default="validation")
    parser.add_argument("--method", choices=("history", "graph", "both"), default="both")
    parser.add_argument("--limit", type=int, help="Use the first N selected examples.")
    parser.add_argument("--run", action="store_true", help="Load models and execute; otherwise preview only.")
    parser.add_argument("--output", type=Path, help="New JSONL file; defaults to tmp/reproduction_<UTC>.jsonl.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    manifest = json.loads((REPORT / "sample_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((REPORT / "summary.json").read_text(encoding="utf-8"))
    indices = manifest["development"] + manifest["validation"] if args.split == "all" else manifest[args.split]
    indices = indices[:args.limit] if args.limit else indices
    methods = {"history": "chat_with_history_rag", "graph": "chat_with_graph_rag"}
    if args.method != "both":
        methods = {args.method: methods[args.method]}
    notebook_bytes = NOTEBOOK.read_bytes()
    cells = selected_cells(json.loads(notebook_bytes))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "tmp" / f"reproduction_{stamp}.jsonl").resolve()
    if output.is_relative_to(REPORT) or output.exists():
        parser.error("Choose a new output file outside the frozen report directory.")
    plan = {
        "split": args.split, "dataset_indices": indices, "methods": list(methods),
        "answer_runs": len(indices) * len(methods), "output": str(output),
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "mode": "run" if args.run else "preview",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print("Preview only. Add --run to load models, check the judge, and generate predictions.")
        return

    os.chdir(ROOT)
    namespace = {"__name__": "notebook_reproduction", "__file__": str(NOTEBOOK)}
    for number, source in enumerate(cells):
        exec(compile(source, f"{NOTEBOOK.name}:selected_cell_{number}", "exec"), namespace)
    if namespace["seed"] != manifest["seed"]:
        raise RuntimeError("Set SEED in .env to the manifest seed before reproduction.")
    dataset_path = Path(namespace["path"])
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_sha256 != frozen["data"]["sha256"]:
        raise RuntimeError("The downloaded dataset differs from the frozen evaluation dataset.")
    if not namespace["JUDGE_OK"]:
        raise RuntimeError("Judge controls failed; predictions were not generated.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "run", **plan, "dataset_sha256": dataset_sha256}) + "\n")
        for method, function in methods.items():
            for index in indices:
                sample = namespace["dataset"][index]
                result = namespace[function]([sample])[0]
                answer = result["answer"]
                message = answer["messages"][-1] if isinstance(answer, dict) else answer
                case = {
                    "type": "abstention" if sample["question_id"].endswith("_abs") else sample["question_type"],
                    "question": sample["question"], "reference": sample["answer"],
                    "candidate": message.content,
                }
                row = {
                    "record_type": "prediction", "method": method, "dataset_index": index,
                    "id": sample["question_id"], **case,
                    "automatic_correct": namespace["grade_answer"](case).correct,
                    "retrieved": result.get("retrieved", []),
                    "seconds": result["exec_time"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"Saved {method}: {row['id']}", flush=True)
    print(f"Saved {output}. Automatic grades require independent review.")


if __name__ == "__main__":
    main()
