"""Dataset loading: RuleTaker, FOLIO, CounterBench and arbitrary test files.

Every example is normalised to a dict with keys
    id, source, context, query, label (bool or None),
    premises_fol, conclusion_fol (optional), meta
"""

import csv
import json
import os
from pathlib import Path


def _default_data_dir() -> Path:
    if os.environ.get("REASONER_DATA_DIR"):
        return Path(os.environ["REASONER_DATA_DIR"])
    try:
        return Path(__file__).resolve().parents[2] / "data" / "raw"
    except NameError:  # running inside the notebook
        for candidate in (Path("data/raw"), Path("../data/raw")):
            if candidate.exists():
                return candidate
        return Path("data/raw")


RAW = _default_data_dir()


def ensure_public_data(raw_dir: Path = RAW) -> None:
    """Download the public datasets from Hugging Face if they are missing."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = {
        "ruletaker": ("tasksource/ruletaker", ("train", "dev", "test")),
        "folio": ("tasksource/folio", ("train", "validation")),
    }
    for short, (repo, splits) in needed.items():
        if all((raw_dir / f"{short}_{s}.jsonl").exists() for s in splits):
            continue
        from datasets import load_dataset

        ds = load_dataset(repo)
        for s in splits:
            ds[s].to_json(str(raw_dir / f"{short}_{s}.jsonl"))
    for name in ("data_balanced_alpha_V1.json", "data_balanced_backdoor_V2.json"):
        target = raw_dir / f"counterbench_{name}"
        if not target.exists():
            import shutil

            from huggingface_hub import hf_hub_download

            path = hf_hub_download("CounterBench/CounterBench", name, repo_type="dataset")
            shutil.copy(path, target)

CONTEXT_KEYS = ("context", "theory", "premises", "given_info", "kb", "knowledge_base",
                "facts", "passage", "story")
QUERY_KEYS = ("statement", "question", "query", "conclusion", "hypothesis", "claim")
LABEL_KEYS = ("flag", "label", "answer", "gold", "target", "truth")
ID_KEYS = ("id", "example_id", "question_id", "uid", "idx")


def to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "yes", "entailment", "entailed", "1", "t", "y"):
        return True
    if v in ("false", "no", "not entailment", "not_entailment", "0", "uncertain", "unknown",
             "f", "n", "contradiction", "neutral"):
        return False
    return None


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ruletaker(split="dev", limit=None, seed=0):
    rows = _read_jsonl(RAW / f"ruletaker_{split}.jsonl")
    if limit:
        import random

        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    return [
        {"id": f"ruletaker-{split}-{i}", "source": "ruletaker", "context": r["context"],
         "query": r["question"], "label": r["label"] == "entailment",
         "meta": {"config": r["config"], "raw_label": r["label"]}}
        for i, r in enumerate(rows)
    ]


def load_folio(split="validation"):
    rows = _read_jsonl(RAW / f"folio_{split}.jsonl")
    return [
        {"id": f"folio-{split}-{r['example_id']}", "source": "folio", "context": r["premises"],
         "query": r["conclusion"], "label": r["label"] == "True",
         "premises_fol": r["premises-FOL"], "conclusion_fol": r["conclusion-FOL"],
         "meta": {"raw_label": r["label"], "story_id": r["story_id"]}}
        for r in rows
    ]


def load_counterbench():
    out = []
    for name in ("data_balanced_alpha_V1", "data_balanced_backdoor_V2"):
        path = RAW / f"counterbench_{name}.json"
        for r in json.loads(path.read_text(encoding="utf-8")):
            out.append({
                "id": f"counterbench-{name}-{r['question_id']}", "source": "counterbench",
                "context": r["given_info"], "query": r["question"],
                "label": r["answer"].strip().lower() == "yes",
                "meta": {"type": r["type"], "raw_label": r["answer"]},
            })
    return out


def _pick(record, keys):
    lowered = {k.lower().strip(): k for k in record}
    for key in keys:
        if key in lowered:
            return record[lowered[key]]
    return None


def _as_text(value, sep=" "):
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        import ast

        try:  # a list serialised into a CSV cell
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if not isinstance(value, str) and hasattr(value, "__iter__"):
        return sep.join(str(v).strip() for v in value)
    return str(value)


def normalize_record(record: dict, index: int) -> dict:
    """Normalise a record from an unseen test file with unknown column names."""
    context = _as_text(_pick(record, CONTEXT_KEYS))
    query = _as_text(_pick(record, QUERY_KEYS))
    rid = _pick(record, ID_KEYS)
    premises_fol = _pick(record, ("premises-fol", "premises_fol", "fol_premises"))
    conclusion_fol = _pick(record, ("conclusion-fol", "conclusion_fol", "fol_conclusion"))
    return {
        "id": str(rid) if rid is not None else str(index),
        "source": str(_pick(record, ("source", "dataset", "config")) or "unknown"),
        "context": context,
        "query": query,
        "label": to_bool(_pick(record, LABEL_KEYS)),
        "premises_fol": _as_text(premises_fol, sep="\n") or None,
        "conclusion_fol": _as_text(conclusion_fol) or None,
        "meta": {},
    }


def load_any(path) -> list[dict]:
    """Load a .csv / .json / .jsonl / .parquet test file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    elif suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data if isinstance(data, list) else data.get("data", [data])
    elif suffix == ".parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    else:
        raise ValueError(f"unsupported file type: {suffix}")
    return [normalize_record(r, i) for i, r in enumerate(rows)]
