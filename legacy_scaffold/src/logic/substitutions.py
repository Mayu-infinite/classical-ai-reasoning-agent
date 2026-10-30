from collections.abc import Mapping

from reasoning_agent.logic.ast import Term

Substitution = dict[str, Term]


def apply_term(term: Term, substitution: Mapping[str, Term]) -> Term:
    """Resolve a variable through a substitution chain."""
    seen: set[str] = set()

    while term.is_variable and term.name in substitution:
        if term.name in seen:
            raise ValueError("Cyclic substitution detected")

        seen.add(term.name)
        term = substitution[term.name]

    return term


def apply_atom(
    atom,
    substitution: Mapping[str, Term],
):
    """Apply substitutions to every argument of an atom."""
    from reasoning_agent.logic.ast import Atom

    return Atom(
        predicate=atom.predicate,
        arguments=tuple(apply_term(term, substitution) for term in atom.arguments),
    )
