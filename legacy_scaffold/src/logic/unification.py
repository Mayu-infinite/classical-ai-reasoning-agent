from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.substitutions import (
    Substitution,
    apply_term,
)


def _occurs(
    variable: str,
    term: Term,
    substitution: Substitution,
) -> bool:
    term = apply_term(term, substitution)
    return term.is_variable and term.name == variable


def unify_terms(
    left: Term,
    right: Term,
    substitution: Substitution | None = None,
) -> Substitution | None:
    """Return a consistent substitution, or None if terms conflict."""
    result = dict(substitution or {})

    left = apply_term(left, result)
    right = apply_term(right, result)

    if left == right:
        return result

    if left.is_variable:
        if _occurs(left.name, right, result):
            return None
        result[left.name] = right
        return result

    if right.is_variable:
        if _occurs(right.name, left, result):
            return None
        result[right.name] = left
        return result

    return None


def unify_atoms(
    left: Atom,
    right: Atom,
    substitution: Substitution | None = None,
) -> Substitution | None:
    """Unify two atoms with the same predicate and arity."""
    if left.predicate != right.predicate:
        return None

    if len(left.arguments) != len(right.arguments):
        return None

    result = dict(substitution or {})

    for left_term, right_term in zip(left.arguments, right.arguments, strict=True):
        unified = unify_terms(left_term, right_term, result)
        if unified is None:
            return None
        result = unified

    return result
