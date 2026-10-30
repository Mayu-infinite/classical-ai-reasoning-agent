from reasoning_agent.inference.backward_chaining import backward_chain
from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import HornRule, KnowledgeBase


def atom(predicate, *arguments):
    return Atom(predicate, tuple(Term(arg) for arg in arguments))


def variable(name):
    return Term(name, is_variable=True)


def test_backward_chaining_proves_query_from_rule():
    kb = KnowledgeBase()
    kb.add_fact(atom("Young", "Bob"))
    kb.add_fact(atom("Kind", "Bob"))

    kb.add_rule(
        HornRule(
            premises=(atom_with_variable("Young", "x"),),
            conclusion=atom_with_variable("Smart", "x"),
        )
    )
    kb.add_rule(
        HornRule(
            premises=(
                atom_with_variable("Smart", "x"),
                atom_with_variable("Kind", "x"),
            ),
            conclusion=atom_with_variable("Successful", "x"),
        )
    )

    assert backward_chain(kb, atom("Successful", "Bob"))


def test_backward_chaining_returns_false_for_unproved_query():
    kb = KnowledgeBase()
    kb.add_fact(atom("Young", "Bob"))

    assert not backward_chain(kb, atom("Successful", "Bob"))


def atom_with_variable(predicate, name):
    return Atom(predicate, (variable(name),))


def test_backward_chaining_propagates_substitutions_across_premises():
    def constant(name: str) -> Term:
        return Term(name)

    def variable(name: str) -> Term:
        return Term(name, is_variable=True)

    kb = KnowledgeBase()

    kb.add_fact(Atom("Parent", (constant("alice"), constant("bob"))))
    kb.add_fact(Atom("Parent", (constant("bob"), constant("carol"))))

    kb.add_rule(
        HornRule(
            premises=(
                Atom("Parent", (variable("x"), variable("y"))),
                Atom("Parent", (variable("y"), variable("z"))),
            ),
            conclusion=Atom("Grandparent", (variable("x"), variable("z"))),
        )
    )

    query = Atom("Grandparent", (constant("alice"), constant("carol")))

    assert backward_chain(kb, query)
    

def test_backward_chaining_backtracks_when_first_match_fails():
    kb = KnowledgeBase()

    kb.add_fact(atom("Parent", "Alice", "Bob"))
    kb.add_fact(atom("Parent", "Alice", "Carol"))
    kb.add_fact(atom("Parent", "Carol", "Dave"))
    kb.add_fact(atom("Eligible", "Dave"))

    kb.add_rule(
        HornRule(
            premises=(
                Atom("Parent", (variable("x"), variable("y"))),
                Atom("Parent", (variable("y"), variable("z"))),
                Atom("Eligible", (variable("z"),)),
            ),
            conclusion=Atom("Grandparent", (variable("x"), variable("z"))),
        )
    )

    assert backward_chain(kb, atom("Grandparent", "Alice", "Dave"))


def test_backward_chaining_terminates_on_recursive_cycle():
    kb = KnowledgeBase()

    kb.add_rule(
        HornRule(
            premises=(Atom("Q", (variable("x"),)),),
            conclusion=Atom("P", (variable("x"),)),
        )
    )
    kb.add_rule(
        HornRule(
            premises=(Atom("P", (variable("x"),)),),
            conclusion=Atom("Q", (variable("x"),)),
        )
    )

    assert not backward_chain(kb, atom("P", "Alice"), max_depth=20)

