
from collections.abc import Iterator
from itertools import count

from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import KnowledgeBase
from reasoning_agent.logic.substitutions import Substitution, apply_atom
from reasoning_agent.logic.unification import unify_atoms


def backward_chain(
    knowledge_base: KnowledgeBase,
    query: Atom,
    *,
    max_depth: int = 100,
) -> bool:
    """Prove a ground query using backward chaining over positive Horn rules."""
    if any(term.is_variable for term in query.arguments):
        raise ValueError("Backward chaining requires a ground query")

    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")

    fresh_counter = count()

    def prove(
        goal: Atom,
        substitution: Substitution,
        depth: int,
        active: frozenset[Atom],
    ) -> Iterator[Substitution]:
        goal = apply_atom(goal, substitution)

        # Try known facts first. Each successful match yields its substitution.
        for fact in knowledge_base.facts:
            match = unify_atoms(goal, fact, substitution)
            if match is not None:
                yield match

        if depth >= max_depth or goal in active:
            return

        next_active = active | {goal}

        for rule in knowledge_base.rules:
            # Standardize this rule's variables apart for every application.
            suffix = next(fresh_counter)
            variable_names = {
                term.name
                for atom in (*rule.premises, rule.conclusion)
                for term in atom.arguments
                if term.is_variable
            }
            rename = {
                name: Term(f"{name}__bc{suffix}", is_variable=True)
                for name in variable_names
            }

            def rename_atom(atom: Atom) -> Atom:
                return Atom(
                    atom.predicate,
                    tuple(
                        rename.get(term.name, term)
                        if term.is_variable
                        else term
                        for term in atom.arguments
                    ),
                )

            conclusion = rename_atom(rule.conclusion)
            unified = unify_atoms(goal, conclusion, substitution)

            if unified is None:
                continue

            def prove_premises(
                index: int,
                current: Substitution,
            ) -> Iterator[Substitution]:
                if index == len(rule.premises):
                    yield current
                    return

                premise = rename_atom(rule.premises[index])

                for result in prove(
                    premise,
                    current,
                    depth + 1,
                    next_active,
                ):
                    yield from prove_premises(index + 1, result)

            yield from prove_premises(0, unified)

    return next(prove(query, {}, 0, frozenset()), None) is not None
