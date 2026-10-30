"""A small, dependency-free propositional logic parser and entailment checker.

Supported operators (precedence high to low): negation (!, ~, NOT, ¬),
conjunction (&, AND, ∧), disjunction (|, OR, ∨), implication (->, =>, →),
and biconditional (<->, <=>, ↔). Atoms are identifiers without arguments.
Entailment is checked semantically by exhaustive truth assignments, so this is
complete for propositional formulas but exponential in the number of symbols.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class PAtom:
    name: str


@dataclass(frozen=True, slots=True)
class PNot:
    operand: PFormula


@dataclass(frozen=True, slots=True)
class PAnd:
    left: PFormula
    right: PFormula


@dataclass(frozen=True, slots=True)
class POr:
    left: PFormula
    right: PFormula


@dataclass(frozen=True, slots=True)
class PImplies:
    left: PFormula
    right: PFormula


@dataclass(frozen=True, slots=True)
class PIff:
    left: PFormula
    right: PFormula


PFormula = PAtom | PNot | PAnd | POr | PImplies | PIff

_TOKEN = re.compile(
    r"\s*(<->|<=>|->|=>|[()!~&|]|¬|∧|∨|→|↔|"
    r"(?i:NOT|AND|OR)\b|[A-Za-z_][A-Za-z0-9_]*)"
)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(text):
        match = _TOKEN.match(text, position)
        if match is None:
            if text[position:].strip() == "":
                break
            raise ValueError(f"Unexpected token near {text[position:position + 20]!r}")
        tokens.append(match.group(1))
        position = match.end()
    if not tokens:
        raise ValueError("Formula cannot be empty")
    return tokens


class _Parser:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens
        self.i = 0

    def peek(self) -> str | None:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of formula")
        self.i += 1
        return token

    @staticmethod
    def norm(token: str | None) -> str:
        return {"¬": "!", "~": "!", "NOT": "!", "∧": "&", "AND": "&",
                "∨": "|", "OR": "|", "→": "->", "=>": "->",
                "↔": "<->", "<=>": "<->"}.get(token.upper() if token else "", token or "")

    def parse(self) -> PFormula:
        result = self.iff()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token {self.peek()!r}")
        return result

    def iff(self) -> PFormula:
        left = self.implies()
        while self.norm(self.peek()) == "<->":
            self.take()
            left = PIff(left, self.implies())
        return left

    def implies(self) -> PFormula:
        left = self.disjunction()
        if self.norm(self.peek()) == "->":
            self.take()
            return PImplies(left, self.implies())
        return left

    def disjunction(self) -> PFormula:
        left = self.conjunction()
        while self.norm(self.peek()) == "|":
            self.take()
            left = POr(left, self.conjunction())
        return left

    def conjunction(self) -> PFormula:
        left = self.unary()
        while self.norm(self.peek()) == "&":
            self.take()
            left = PAnd(left, self.unary())
        return left

    def unary(self) -> PFormula:
        token = self.peek()
        if self.norm(token) == "!":
            self.take()
            return PNot(self.unary())
        if token == "(":
            self.take()
            result = self.iff()
            if self.take() != ")":
                raise ValueError("Expected ')'")
            return result
        if token is None or token in {")", "&", "|", "->", "<->", "∧", "∨", "→", "↔"}:
            raise ValueError(f"Expected a proposition, found {token!r}")
        self.take()
        if token.upper() in {"TRUE", "FALSE"}:
            return PAtom(token.upper())
        return PAtom(token)


def parse_formula(text: str) -> PFormula:
    """Parse a propositional formula."""
    return _Parser(_tokens(text)).parse()


def symbols(formula: PFormula) -> set[str]:
    if isinstance(formula, PAtom):
        return set() if formula.name in {"TRUE", "FALSE"} else {formula.name}
    if isinstance(formula, PNot):
        return symbols(formula.operand)
    if isinstance(formula, (PAnd, POr, PImplies, PIff)):
        return symbols(formula.left) | symbols(formula.right)
    raise TypeError(f"Unsupported formula node: {type(formula).__name__}")


def evaluate(formula: PFormula, assignment: dict[str, bool]) -> bool:
    if isinstance(formula, PAtom):
        if formula.name == "TRUE":
            return True
        if formula.name == "FALSE":
            return False
        return assignment[formula.name]
    if isinstance(formula, PNot):
        return not evaluate(formula.operand, assignment)
    if isinstance(formula, PAnd):
        return evaluate(formula.left, assignment) and evaluate(formula.right, assignment)
    if isinstance(formula, POr):
        return evaluate(formula.left, assignment) or evaluate(formula.right, assignment)
    if isinstance(formula, PImplies):
        return not evaluate(formula.left, assignment) or evaluate(formula.right, assignment)
    if isinstance(formula, PIff):
        return evaluate(formula.left, assignment) == evaluate(formula.right, assignment)
    raise TypeError(f"Unsupported formula node: {type(formula).__name__}")


def entails(premises: list[str] | tuple[str, ...], query: str, *, max_symbols: int = 20) -> bool:
    """Return True iff premises semantically entail query.

    Raises ValueError instead of silently returning a guess if the problem has
    more than ``max_symbols`` distinct propositional symbols.
    """
    formulas = [parse_formula(item) for item in premises]
    target = parse_formula(query)
    names = set().union(*(symbols(f) for f in [*formulas, target]))
    if len(names) > max_symbols:
        raise ValueError(f"{len(names)} symbols exceeds max_symbols={max_symbols}")
    ordered = sorted(names)
    for values in product((False, True), repeat=len(ordered)):
        assignment = dict(zip(ordered, values, strict=True))
        if all(evaluate(f, assignment) for f in formulas) and not evaluate(target, assignment):
            return False
    return True
