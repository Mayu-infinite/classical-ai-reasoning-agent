from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.unification import unify_atoms, unify_terms


def test_variable_unifies_with_constant():
    result = unify_terms(
        Term("x", is_variable=True),
        Term("Bob"),
    )

    assert result == {"x": Term("Bob")}


def test_different_constants_do_not_unify():
    result = unify_terms(Term("Bob"), Term("Alice"))

    assert result is None


def test_same_predicate_unifies():
    left = Atom("Young", (Term("x", is_variable=True),))
    right = Atom("Young", (Term("Bob"),))

    result = unify_atoms(left, right)

    assert result == {"x": Term("Bob")}


def test_different_predicates_do_not_unify():
    left = Atom("Young", (Term("Bob"),))
    right = Atom("Kind", (Term("Bob"),))

    assert unify_atoms(left, right) is None


def test_different_arities_do_not_unify():
    left = Atom("Knows", (Term("Bob"),))
    right = Atom("Knows", (Term("Bob"), Term("Alice")))

    assert unify_atoms(left, right) is None


def test_occurs_check_rejects_self_reference():
    result = unify_terms(
        Term("x", is_variable=True),
        Term("x", is_variable=True),
    )

    assert result == {}
