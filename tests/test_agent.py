import pytest

from reasoning_agent.agent import ReasoningAgent
from reasoning_agent.data import normalize_record, to_bool

ASSIGNMENT_KB = ("Bob is young. Bob is kind. If someone is young, then they are smart. "
                 "If someone is smart and kind, then they are successful.")


@pytest.mark.parametrize("strategy", ["forward", "backward"])
def test_assignment_example(strategy):
    agent = ReasoningAgent(strategy=strategy)
    assert agent.predict(ASSIGNMENT_KB, "Bob is successful.").label is True
    assert agent.predict(ASSIGNMENT_KB, "Bob is not successful.").label is False
    assert agent.predict(ASSIGNMENT_KB, "Alice is successful.").label is False


def test_closed_world_negative_query():
    ctx = "The cat is big. If something is big and not red then it is cold."
    agent = ReasoningAgent()
    assert agent.predict(ctx, "The cat is cold.").label is True
    assert agent.predict(ctx, "The cat is not red.").label is True


def test_fol_annotations_use_resolution():
    pred = ReasoningAgent().predict(
        "ignored", "ignored",
        premises_fol="∀x (Human(x) → Mortal(x))\nHuman(socrates)",
        conclusion_fol="Mortal(socrates)")
    assert pred.engine == "resolution" and pred.label is True


def test_causal_routing():
    pred = ReasoningAgent().predict(
        "We know that A causes B, and B causes C.", "Would C occur if not A instead of A?")
    assert pred.engine == "causal" and pred.label is False


def test_free_form_text_is_open_world():
    agent = ReasoningAgent()
    ctx = "All dogs are animals. Rex is a dog. James is either a teacher or a doctor."
    assert agent.predict(ctx, "Rex is an animal.").label is True
    # Not stated and not derivable: unknown, hence not entailed.
    assert agent.predict(ctx, "James is not a teacher.").label is False


def test_explanations_are_produced():
    pred = ReasoningAgent().predict(ASSIGNMENT_KB, "Bob is successful.", explain_proof=True)
    assert any("rule" in line for line in pred.trace)


def test_record_normalisation():
    rec = normalize_record({"ID": 7, "Context": "Bob is big.", "Statement": "Bob is big.",
                            "Flag": "True"}, 0)
    assert rec["id"] == "7" and rec["label"] is True
    assert to_bool("not entailment") is False and to_bool("yes") is True


@pytest.mark.parametrize("fol", [["∀x (Man(x) → Mortal(x))", "Man(socrates)"],
                                 "['∀x (Man(x) → Mortal(x))', 'Man(socrates)']"])
def test_list_valued_fol_columns(fol):
    rec = normalize_record({"premises": ["All men are mortal.", "Socrates is a man."],
                            "conclusion": "Socrates is mortal.", "premises-FOL": fol,
                            "conclusion-FOL": "Mortal(socrates)"}, 0)
    pred = ReasoningAgent().predict(rec["context"], rec["query"], rec["premises_fol"],
                                    rec["conclusion_fol"])
    assert pred.engine == "resolution" and pred.label is True
