import pytest

from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import HornRule, KnowledgeBase
from reasoning_agent.pipeline import predict_from_logic


def test_pipeline_predicts_true_for_entailed_query():
    bob = Term("Bob")
    x = Term("x", is_variable=True)

    kb = KnowledgeBase(
        facts={Atom("Young", (bob,))},
        rules=[
            HornRule(
                premises=(Atom("Young", (x,)),),
                conclusion=Atom("Smart", (x,)),
            )
        ],
    )

    assert (
        predict_from_logic(
            kb,
            Atom("Smart", (bob,)),
        )
        is True
    )


def test_pipeline_returns_false_for_underived_query():
    kb = KnowledgeBase(facts={Atom("Young", (Term("Bob"),))})

    assert (
        predict_from_logic(
            kb,
            Atom("Successful", (Term("Bob"),)),
        )
        is False
    )


def test_engine_rejects_variable_query():
    kb = KnowledgeBase()

    with pytest.raises(ValueError):
        predict_from_logic(
            kb,
            Atom("Young", (Term("x", is_variable=True),)),
        )
