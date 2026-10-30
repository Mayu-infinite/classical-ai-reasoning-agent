"""Normalization helpers for CounterBench records.

CounterBench questions may be counterfactual or require causal/world knowledge.
They are preserved as task records and are not silently converted to ordinary
monotonic entailment questions.
"""
from typing import Any


def adapt_counterbench_record(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a CounterBench record while preserving its task metadata."""
    question = record.get("question")
    given_info = record.get("given_info")
    answer = record.get("answer")
    if not isinstance(question, str) or not isinstance(given_info, str):
        raise ValueError("CounterBench record requires question and given_info text")
    if answer is None:
        raise ValueError("CounterBench record is missing its answer")
    return {
        "id": str(record.get("question_id", "")),
        "question": question,
        "given_info": given_info,
        "answer": str(answer),
        "type": str(record.get("type", "unknown")),
        "metadata": record.get("meta", {}),
    }
