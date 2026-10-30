import pytest

from reasoning_agent.logic.propositional import entails, evaluate, parse_formula


def test_modus_ponens():
    assert entails(["P", "P -> Q"], "Q")


def test_disjunction_elimination():
    assert entails(["P | Q", "!P"], "Q")


def test_does_not_infer_from_insufficient_premises():
    assert not entails(["P -> Q"], "Q")


def test_biconditional_and_unicode_operators():
    assert entails(["P ↔ Q", "P"], "Q")


def test_parentheses_and_precedence():
    assert evaluate(parse_formula("P | Q & !R"), {"P": False, "Q": True, "R": False})


def test_symbol_limit_is_explicit():
    with pytest.raises(ValueError, match="max_symbols"):
        entails(["A & B & C"], "A", max_symbols=2)


def test_malformed_formula_is_rejected():
    with pytest.raises(ValueError):
        parse_formula("P & -> Q")
