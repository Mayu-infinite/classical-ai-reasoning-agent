from reasoning_agent.logic.ast import Atom
from reasoning_agent.logic.knowledge_base import KnowledgeBase
from reasoning_agent.logic.substitutions import apply_atom
from reasoning_agent.logic.unification import unify_atoms


def _match_premises(
    premises: tuple[Atom, ...],
    facts: set[Atom],
    index: int = 0,
    substitution: dict | None = None,
):
    """Yield substitutions that satisfy all rule premises."""
    substitution = dict(substitution or {})

    if index == len(premises):
        yield substitution
        return

    premise = apply_atom(premises[index], substitution)

    # Use a snapshot: the caller may add derived facts to the original set.
    for fact in tuple(facts):
        unified = unify_atoms(premise, fact, substitution)

        if unified is not None:
            yield from _match_premises(
                premises,
                facts,
                index + 1,
                unified,
            )


def forward_chain(
    knowledge_base: KnowledgeBase,
    *,
    max_facts: int = 100_000,
) -> set[Atom]:
    """Compute the closure of a finite positive Horn knowledge base."""
    if max_facts < 0:
        raise ValueError("max_facts cannot be negative")

    facts = set(knowledge_base.facts)

    if len(facts) > max_facts:
        raise RuntimeError("Initial knowledge base exceeds max_facts")

    changed = True

    while changed:
        changed = False
        new_facts: set[Atom] = set()

        # Collect new facts separately during this iteration.
        for rule in knowledge_base.rules:
            for substitution in _match_premises(
                rule.premises,
                facts,
            ):
                conclusion = apply_atom(
                    rule.conclusion,
                    substitution,
                )

                # Only ground conclusions are supported by this engine.
                if any(term.is_variable for term in conclusion.arguments):
                    continue

                if conclusion not in facts:
                    new_facts.add(conclusion)

        if new_facts:
            if len(facts) + len(new_facts) > max_facts:
                raise RuntimeError("Inference exceeded max_facts")

            facts.update(new_facts)
            changed = True

    return facts
