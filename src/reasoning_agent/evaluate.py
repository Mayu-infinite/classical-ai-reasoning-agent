"""Evaluation helpers: accuracy, timing, engine usage, and submission files."""

import csv
import time
from collections import Counter, defaultdict

from reasoning_agent.agent import ReasoningAgent


def run(agent: ReasoningAgent, examples, use_fol=True, progress_every=0):
    """Predict every example. Returns a list of (example, Prediction)."""
    out = []
    for i, ex in enumerate(examples, 1):
        pred = agent.predict(
            ex["context"], ex["query"],
            ex.get("premises_fol") if use_fol else None,
            ex.get("conclusion_fol") if use_fol else None,
        )
        out.append((ex, pred))
        if progress_every and i % progress_every == 0:
            print(f"  {i}/{len(examples)}")
    return out


def summarize(results, title=""):
    by_source = defaultdict(Counter)
    engines = Counter()
    total_time = 0.0
    for ex, pred in results:
        total_time += pred.seconds
        engines[pred.engine] += 1
        if ex["label"] is None:
            continue
        key = ex["source"]
        by_source[key]["n"] += 1
        by_source[key]["correct"] += int(pred.label == ex["label"])
    n = sum(c["n"] for c in by_source.values())
    correct = sum(c["correct"] for c in by_source.values())
    lines = [f"== {title} ==" if title else "== results =="]
    for source, c in sorted(by_source.items()):
        lines.append(f"{source:>14}: {c['correct']}/{c['n']} = {100 * c['correct'] / c['n']:.2f}%")
    if n:
        lines.append(f"{'overall':>14}: {correct}/{n} = {100 * correct / n:.2f}%")
    lines.append(f"total inference time: {total_time:.3f}s "
                 f"({1000 * total_time / max(1, len(results)):.3f} ms/example)")
    lines.append("engines: " + ", ".join(f"{k}={v}" for k, v in engines.most_common()))
    return "\n".join(lines), (100 * correct / n if n else None), total_time


def breakdown(results, key):
    table = defaultdict(Counter)
    for ex, pred in results:
        if ex["label"] is None:
            continue
        k = key(ex)
        table[k]["n"] += 1
        table[k]["correct"] += int(pred.label == ex["label"])
    return {k: (v["correct"], v["n"]) for k, v in sorted(table.items())}


def write_submission(results, path="submission.csv", label_column="label"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", label_column])
        for ex, pred in results:
            writer.writerow([ex["id"], "True" if pred.label else "False"])


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter() - start
