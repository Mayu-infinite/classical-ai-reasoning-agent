"""Normalization helpers for the public FOLIO dataset.

FOLIO's FOL annotations are retained verbatim. This module deliberately does
not pretend that the propositional solver understands quantified first-order
logic; unsupported syntax must be reported by the caller.
"""
from typing import Any


def adapt_folio_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tasksource/folio record without losing its FOL annotation."""
    premises = record.get("premises-FOL")
    conclusion = record.get("conclusion-FOL")
    if not isinstance(premises, str) or not isinstance(conclusion, str):
        raise ValueError("FOLIO record is missing premises-FOL or conclusion-FOL")
    raw_label = str(record.get("label", "")).strip().lower()
    labels = {"true": "entailed", "false": "contradicted", "unknown": "unknown"}
    if raw_label not in labels:
        raise ValueError(f"Unsupported FOLIO label: {raw_label!r}")
    return {
        "id": str(record.get("example_id", record.get("story_id", ""))),
        "premises_fol": premises,
        "conclusion_fol": conclusion,
        "label": labels[raw_label],
        "premises_nl": record.get("premises", ""),
        "conclusion_nl": record.get("conclusion", ""),
    }
