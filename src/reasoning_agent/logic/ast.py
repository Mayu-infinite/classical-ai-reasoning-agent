from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Term:
    """A constant or variable."""

    name: str
    is_variable: bool = False


@dataclass(frozen=True, slots=True)
class Atom:
    """A predicate applied to zero or more terms."""

    predicate: str
    arguments: tuple[Term, ...] = ()


@dataclass(frozen=True, slots=True)
class Not:
    operand: Formula


@dataclass(frozen=True, slots=True)
class And:
    operands: tuple[Formula, ...]


@dataclass(frozen=True, slots=True)
class Or:
    operands: tuple[Formula, ...]


@dataclass(frozen=True, slots=True)
class Implies:
    antecedent: Formula
    consequent: Formula


@dataclass(frozen=True, slots=True)
class ForAll:
    variables: tuple[str, ...]
    body: Formula


@dataclass(frozen=True, slots=True)
class Exists:
    variables: tuple[str, ...]
    body: Formula


Formula = Atom | Not | And | Or | Implies | ForAll | Exists
