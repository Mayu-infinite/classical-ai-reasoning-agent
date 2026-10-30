
import pytest

from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import HornRule, KnowledgeBase
from reasoning_agent.pipeline import predict_from_logic


def atom(predicate: str, *arguments: str) -> Atom:
    """Create an atom containing constant arguments."""
    return Atom(
        predicate,
        tuple(Term(argument) for argument in arguments),
    )


def variable(name: str) -> Term:
    """Create a variable term."""
    return Term(name, is_variable=True)


def test_pipeline_predicts_true_for_entailed_query():
    bob = Term("Bob")
    x = variable("x")

    kb = KnowledgeBase(
        facts={Atom("Young", (bob,))},
        rules=[
            HornRule(
                premises=(Atom("Young", (x,)),),
                conclusion=Atom("Smart", (x,)),
            )
        ],
    )

    assert predict_from_logic(kb, Atom("Smart", (bob,))) is True


def test_pipeline_returns_false_for_underived_query():
    kb = KnowledgeBase(facts={Atom("Young", (Term("Bob"),))})

    assert predict_from_logic(
        kb,
        Atom("Successful", (Term("Bob"),)),
    ) is False


def test_engine_rejects_variable_query():
    kb = KnowledgeBase()

    with pytest.raises(ValueError):
        predict_from_logic(
            kb,
            Atom("Young", (variable("x"),)),
        )


def test_pipeline_supports_backward_chaining():
    kb = KnowledgeBase(
        facts={atom("Parent", "Alice", "Bob")},
        rules=[
            HornRule(
                premises=(
                    Atom("Parent", (variable("x"), variable("y"))),
                ),
                conclusion=Atom("HasParent", (variable("y"),)),
            )
        ],
    )

    assert predict_from_logic(
        kb,
        atom("HasParent", "Bob"),
        strategy="backward",
    ) is True


def test_pipeline_rejects_unknown_inference_strategy():
    kb = KnowledgeBase()

    with pytest.raises(ValueError, match="Unknown inference strategy"):
        predict_from_logic(
            kb,
            atom("Young", "Bob"),
            strategy="invalid",
        )
