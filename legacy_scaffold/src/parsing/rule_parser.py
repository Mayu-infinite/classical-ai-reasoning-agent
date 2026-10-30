
import re

from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import HornRule, KnowledgeBase

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_atom(text: str) -> Atom:
    """Parse an atom such as Parent(Alice, Bob) or Ready."""
    text = text.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\((.*)\))?", text)

    if match is None:
        raise ValueError(f"Invalid atom: {text!r}")

    predicate, raw_arguments = match.groups()

    if raw_arguments is None or not raw_arguments.strip():
        return Atom(predicate)

    arguments = []
    for raw_argument in raw_arguments.split(","):
        name = raw_argument.strip()

        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"Invalid term: {name!r}")

        is_variable = name[0].islower()
        arguments.append(Term(name, is_variable=is_variable))

    return Atom(predicate, tuple(arguments))


def parse_rule(text: str) -> HornRule:
    """Parse a positive Horn rule: A(x) & B(x) -> C(x)."""
    parts = text.split("->")

    if len(parts) != 2:
        raise ValueError(f"Expected exactly one '->' in rule: {text!r}")

    raw_premises, raw_conclusion = parts
    premise_texts = raw_premises.split("&")

    if not all(part.strip() for part in premise_texts):
        raise ValueError(f"Rule contains an empty premise: {text!r}")

    premises = tuple(parse_atom(part) for part in premise_texts)
    conclusion = parse_atom(raw_conclusion)

    return HornRule(premises=premises, conclusion=conclusion)


def parse_knowledge_base(
    facts: list[str],
    rules: list[str],
) -> KnowledgeBase:
    """Build a knowledge base from fact and rule strings."""
    knowledge_base = KnowledgeBase()

    for fact_text in facts:
        fact = parse_atom(fact_text)

        if any(term.is_variable for term in fact.arguments):
            raise ValueError(f"Facts must be ground: {fact_text!r}")

        knowledge_base.add_fact(fact)

    for rule_text in rules:
        knowledge_base.add_rule(parse_rule(rule_text))

    return knowledge_base
