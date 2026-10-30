import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSON Lines file."""
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")

            yield record


def load_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file containing records or a record collection."""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return [data]

    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data

    raise ValueError(f"Expected a JSON object or list of objects: {path}")


def load_csv(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows as dictionaries."""
    with Path(path).open(encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def load_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Dispatch based on the file extension."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        yield from load_jsonl(path)
    elif suffix == ".json":
        yield from load_json(path)
    elif suffix == ".csv":
        yield from load_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
