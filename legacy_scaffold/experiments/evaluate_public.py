
import argparse
import json
import time
from pathlib import Path
from typing import Any

from reasoning_agent.inference.engine import ReasoningEngine
from reasoning_agent.logic.ast import Atom
from reasoning_agent.parsing.rule_parser import (
    parse_atom,
    parse_knowledge_base,
)


def load_examples(path: Path) -> list[dict[str, Any]]:
    """Load labeled evaluation records from JSONL or JSON."""
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Expected an object on line {line_number}"
                    )
                records.append(record)
        return records

    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list) and all(
            isinstance(item, dict) for item in data
        ):
            return data
        raise ValueError("JSON must contain an object or a list of objects")

    raise ValueError("Supported formats: .jsonl and .json")


def parse_label(value: Any) -> bool:
    """Convert a supported Boolean label to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise ValueError(f"Unsupported label: {value!r}")


def evaluate_record(
    record: dict[str, Any],
    engine: ReasoningEngine,
    strategy: str,
) -> tuple[bool, float]:
    """Evaluate one example and return (correct, elapsed_seconds)."""
    facts = record.get("facts", [])
    rules = record.get("rules", [])
    query_text = record.get("query")
    expected = parse_label(record.get("label"))

    if not isinstance(facts, list) or not all(
        isinstance(item, str) for item in facts
    ):
        raise ValueError("'facts' must be a list of strings")

    if not isinstance(rules, list) or not all(
        isinstance(item, str) for item in rules
    ):
        raise ValueError("'rules' must be a list of strings")

    if not isinstance(query_text, str):
        raise ValueError("'query' must be a string")

    knowledge_base = parse_knowledge_base(facts, rules)
    query: Atom = parse_atom(query_text)

    start = time.perf_counter()
    prediction = engine.entails(
        knowledge_base,
        query,
        strategy=strategy,
    )
    elapsed = time.perf_counter() - start

    return prediction == expected, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the classical reasoning engine."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to a labeled .json or .jsonl evaluation file",
    )
    parser.add_argument(
        "--strategy",
        choices=("forward", "backward"),
        default="forward",
    )
    args = parser.parse_args()

    records = load_examples(args.dataset)
    if not records:
        raise ValueError("The evaluation dataset is empty")

    engine = ReasoningEngine()
    correct = 0
    total_time = 0.0
    evaluated = 0

    for index, record in enumerate(records, start=1):
        try:
            is_correct, elapsed = evaluate_record(
                record,
                engine,
                args.strategy,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(
                f"Invalid example at record {index}: {exc}"
            ) from exc

        correct += int(is_correct)
        total_time += elapsed
        evaluated += 1

    accuracy = correct / evaluated
    average_time = total_time / evaluated

    print(f"Dataset: {args.dataset}")
    print(f"Strategy: {args.strategy}")
    print(f"Examples evaluated: {evaluated}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Total inference time: {total_time:.6f} seconds")
    print(f"Average inference time: {average_time:.6f} seconds")


if __name__ == "__main__":
    main()
