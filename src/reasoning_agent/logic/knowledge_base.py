from dataclasses import dataclass, field

from reasoning_agent.logic.ast import Atom


@dataclass(frozen=True, slots=True)
class HornRule:
    """A conjunction of positive premises implying one positive atom."""

    premises: tuple[Atom, ...]
    conclusion: Atom


@dataclass(slots=True)
class KnowledgeBase:
    """Facts and Horn rules used by the initial inference engine."""

    facts: set[Atom] = field(default_factory=set)
    rules: list[HornRule] = field(default_factory=list)

    def add_fact(self, fact: Atom) -> None:
        self.facts.add(fact)

    def add_rule(self, rule: HornRule) -> None:
        self.rules.append(rule)
