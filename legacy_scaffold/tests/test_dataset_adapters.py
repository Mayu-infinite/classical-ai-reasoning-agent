import pytest

from reasoning_agent.data.counterbench import adapt_counterbench_record
from reasoning_agent.data.folio import adapt_folio_record


def test_folio_adapter_preserves_fol_and_three_way_label():
    row = adapt_folio_record({
        "example_id": 7,
        "premises-FOL": "P(a) -> Q(a)",
        "conclusion-FOL": "Q(a)",
        "label": "True",
        "premises": "premise text",
        "conclusion": "conclusion text",
    })
    assert row["id"] == "7"
    assert row["label"] == "entailed"
    assert row["premises_fol"] == "P(a) -> Q(a)"


def test_folio_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unsupported FOLIO label"):
        adapt_folio_record({"premises-FOL": "P", "conclusion-FOL": "Q", "label": "maybe"})


def test_counterbench_adapter_preserves_task_type():
    row = adapt_counterbench_record({
        "question_id": "cb-1", "question": "What if P?", "given_info": "P is true.",
        "answer": "Q", "type": "counterfactual", "meta": {"source": "test"},
    })
    assert row["id"] == "cb-1"
    assert row["type"] == "counterfactual"
    assert row["metadata"]["source"] == "test"
