
import pytest

from reasoning_agent.parsing.rule_parser import (
    parse_atom,
    parse_knowledge_base,
    parse_rule,
)


def test_parse_ground_atom():
    atom = parse_atom("Parent(Alice, Bob)")

    assert atom.predicate == "Parent"
    assert [term.name for term in atom.arguments] == ["Alice", "Bob"]
    assert all(not term.is_variable for term in atom.arguments)


def test_parse_rule_with_two_premises():
    rule = parse_rule("Parent(x, y) & Parent(y, z) -> Grandparent(x, z)")

    assert len(rule.premises) == 2
    assert rule.conclusion.predicate == "Grandparent"
    assert rule.premises[0].arguments[0].is_variable


def test_parse_knowledge_base_rejects_variable_fact():
    with pytest.raises(ValueError, match="Facts must be ground"):
        parse_knowledge_base(["Parent(x, Bob)"], [])


def test_parse_rule_rejects_missing_arrow():
    with pytest.raises(ValueError, match="exactly one"):
        parse_rule("Parent(Alice, Bob)")
