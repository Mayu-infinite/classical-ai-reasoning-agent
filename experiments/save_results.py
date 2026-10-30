"""Evaluate the agent on the public datasets and save every result as CSV.

Writes to results/ (or --out):
    main.csv        accuracy and inference time per dataset, plus an overall row
    breakdown.csv   accuracy per RuleTaker depth, FOLIO gold label and CounterBench type
    errors.csv      every misclassified query, with the engine that answered it
    ablations.csv   design comparisons from experiments/ablations.py
    run_info.json   when and where the results were produced

Usage:
    uv run python experiments/save_results.py [--skip-ablations] [--ruletaker-split test]
    uv run python experiments/make_figures.py
"""

import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

from ablations import run_ablations

from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.data import load_counterbench, load_folio, load_ruletaker
from reasoning_agent.evaluate import breakdown, run

ROOT = Path(__file__).resolve().parents[1]


def shown(path: Path) -> Path:
    path = path.resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {shown(path)}")


def accuracy_row(dataset, correct, total, seconds, queries):
    return {"dataset": dataset, "correct": correct, "total": total,
            "accuracy": round(100 * correct / total, 2) if total else None,
            "seconds": round(seconds, 3),
            "ms_per_query": round(1000 * seconds / queries, 4) if queries else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--ruletaker-split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = [
        (f"RuleTaker {args.ruletaker_split}", lambda: load_ruletaker(args.ruletaker_split),
         "depth", lambda e: e["meta"]["config"]),
        ("FOLIO validation", lambda: load_folio("validation"),
         "gold label", lambda e: e["meta"]["raw_label"]),
        ("FOLIO train", lambda: load_folio("train"),
         "gold label", lambda e: e["meta"]["raw_label"]),
        ("CounterBench", load_counterbench, "question type", lambda e: e["meta"]["type"]),
    ]

    agent = ReasoningAgent()
    main_rows, breakdown_rows, error_rows = [], [], []
    totals = {"correct": 0, "total": 0, "seconds": 0.0, "queries": 0}
    started = time.perf_counter()
    for name, loader, field, key in datasets:
        print(f"evaluating {name} ...", flush=True)
        results = run(agent, loader())
        labelled = [(ex, pred) for ex, pred in results if ex["label"] is not None]
        correct = sum(pred.label == ex["label"] for ex, pred in labelled)
        seconds = sum(pred.seconds for _, pred in results)
        main_rows.append(accuracy_row(name, correct, len(labelled), seconds, len(results)))
        print(f"  {correct}/{len(labelled)} correct, {seconds:.2f} s", flush=True)
        totals["correct"] += correct
        totals["total"] += len(labelled)
        totals["seconds"] += seconds
        totals["queries"] += len(results)

        for group, (c, n) in breakdown(results, key).items():
            breakdown_rows.append({"dataset": name, "field": field, "group": group,
                                   "correct": c, "total": n,
                                   "accuracy": round(100 * c / n, 2)})
        for ex, pred in labelled:
            if pred.label != ex["label"]:
                error_rows.append({"dataset": name, "id": ex["id"], "gold": ex["label"],
                                   "raw_label": ex["meta"].get("raw_label"),
                                   "predicted": pred.label, "engine": pred.engine,
                                   "seconds": round(pred.seconds, 4),
                                   "query": ex["query"], "context": ex["context"]})

    main_rows.append(accuracy_row("Overall", totals["correct"], totals["total"],
                                  totals["seconds"], totals["queries"]))
    write_csv(args.out / "main.csv", main_rows)
    write_csv(args.out / "breakdown.csv", breakdown_rows)
    if error_rows:
        write_csv(args.out / "errors.csv", error_rows)

    if not args.skip_ablations:
        print("\nrunning ablations ...", flush=True)
        write_csv(args.out / "ablations.csv", list(run_ablations()))

    info = {"generated": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "ruletaker_split": args.ruletaker_split,
            "ablations": not args.skip_ablations,
            "wall_clock_seconds": round(time.perf_counter() - started, 1)}
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {shown(args.out / 'run_info.json')}")


if __name__ == "__main__":
    main()
