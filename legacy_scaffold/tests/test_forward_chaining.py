from reasoning_agent.inference.forward_chaining import forward_chain
from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import HornRule, KnowledgeBase


def constant(name: str) -> Term:
    return Term(name)


def variable(name: str) -> Term:
    return Term(name, is_variable=True)


def atom(predicate: str, *arguments: Term) -> Atom:
    return Atom(predicate, tuple(arguments))


def test_forward_chaining_derives_successful():
    kb = KnowledgeBase()

    kb.add_fact(atom("Young", constant("Bob")))
    kb.add_fact(atom("Kind", constant("Bob")))

    kb.add_rule(
        HornRule(
            premises=(atom("Young", variable("x")),),
            conclusion=atom("Smart", variable("x")),
        )
    )

    kb.add_rule(
        HornRule(
            premises=(
                atom("Smart", variable("x")),
                atom("Kind", variable("x")),
            ),
            conclusion=atom("Successful", variable("x")),
        )
    )

    derived = forward_chain(kb)

    assert atom("Smart", constant("Bob")) in derived
    assert atom("Successful", constant("Bob")) in derived


def test_forward_chaining_preserves_existing_facts():
    fact = atom("Young", constant("Bob"))
    kb = KnowledgeBase(facts={fact})

    assert fact in forward_chain(kb)


def test_empty_knowledge_base_returns_no_facts():
    assert forward_chain(KnowledgeBase()) == set()


def test_max_facts_limit():
    kb = KnowledgeBase(
        facts={
            atom("Young", constant("Bob")),
            atom("Kind", constant("Bob")),
        }
    )

    try:
        forward_chain(kb, max_facts=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected the fact limit to be enforced")
