"""First-order logic representation, a tolerant FOL parser, and CNF conversion.

Terms
    Var            -- a logical variable
    str            -- a constant symbol
    tuple          -- a function application ``(name, arg1, arg2, ...)``

Formulas are nested tuples:
    ("atom", pred, args)      ("not", f)          ("and", f1, f2, ...)
    ("or", f1, f2, ...)       ("imp", a, b)       ("iff", a, b)
    ("xor", a, b)             ("all", var, f)     ("ex", var, f)
    ("true",)                 ("false",)

Literals used by the provers are ``(positive, pred, args)`` triples and a
clause is a tuple of literals (a disjunction).
"""

import itertools
import re


class Var:
    """A logical variable. Identity is the integer id, the name is cosmetic."""

    __slots__ = ("id", "name")
    _counter = itertools.count()

    def __init__(self, name: str = "v"):
        self.id = next(Var._counter)
        self.name = name

    def __repr__(self) -> str:
        return f"{self.name.upper()}{self.id}"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return self.id

    def __lt__(self, other):
        return str(self) < str(other)


def is_var(term) -> bool:
    return type(term) is Var


def term_str(term) -> str:
    if type(term) is tuple:
        return f"{term[0]}({', '.join(term_str(a) for a in term[1:])})"
    return str(term)


def atom_str(pred: str, args: tuple) -> str:
    if pred == "=" and len(args) == 2:
        return f"{term_str(args[0])} = {term_str(args[1])}"
    if not args:
        return pred
    return f"{pred}({', '.join(term_str(a) for a in args)})"


def literal_str(lit) -> str:
    positive, pred, args = lit
    return ("" if positive else "¬") + atom_str(pred, args)


def clause_str(clause) -> str:
    """Print a clause with its variables renamed X, Y, Z, ... for readability."""
    if not clause:
        return "□"
    names: dict[Var, str] = {}

    def rename(t):
        if type(t) is Var:
            if t not in names:
                i = len(names)
                names[t] = "XYZWUV"[i] if i < 6 else f"X{i}"
            return names[t]
        if type(t) is tuple:
            return (t[0], *(rename(a) for a in t[1:]))
        return t

    return " ∨ ".join(
        literal_str((pos, pred, tuple(rename(a) for a in args))) for pos, pred, args in clause
    )


def formula_str(f) -> str:
    op = f[0]
    if op == "atom":
        return atom_str(f[1], f[2])
    if op == "true":
        return "⊤"
    if op == "false":
        return "⊥"
    if op == "not":
        return "¬" + formula_str(f[1])
    if op in ("all", "ex"):
        q = "∀" if op == "all" else "∃"
        return f"{q}{f[1]!r} {formula_str(f[2])}"
    symbol = {"and": " ∧ ", "or": " ∨ ", "imp": " → ", "iff": " ↔ ", "xor": " ⊕ "}[op]
    return "(" + symbol.join(formula_str(g) for g in f[1:]) + ")"


# ---------------------------------------------------------------------------
# Formula constructors
# ---------------------------------------------------------------------------


def Atom(pred: str, *args):
    return ("atom", pred, tuple(args))


def Not(f):
    return ("not", f)


def And(*fs):
    fs = [f for f in fs if f != ("true",)]
    if not fs:
        return ("true",)
    return fs[0] if len(fs) == 1 else ("and", *fs)


def Or(*fs):
    fs = [f for f in fs if f != ("false",)]
    if not fs:
        return ("false",)
    return fs[0] if len(fs) == 1 else ("or", *fs)


def Implies(a, b):
    return ("imp", a, b)


def ForAll(var, f):
    return ("all", var, f)


def Exists(var, f):
    return ("ex", var, f)


# ---------------------------------------------------------------------------
# Tolerant parser for FOLIO-style first-order logic strings
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(?P<op><->|<=>|[-—–]>|=>|⟷|↔|⇔|→|⇒|⊕|∧|∨|¬|∀|∃|≠|!=|=|&|\||~|\^|\(|\)|,|\[|\]|\{|\})"
    r"|(?P<id>\w[\w'.\-]*)"
    r"|(?P<junk>\S))"
)

_OP_ALIASES = {
    "<->": "↔", "<=>": "↔", "⟷": "↔", "⇔": "↔", "->": "→", "—>": "→", "–>": "→",
    "=>": "→", "⇒": "→", "&": "∧", "^": "∧",
    "|": "∨", "~": "¬", "!=": "≠", "[": "(", "]": ")", "{": "(", "}": ")",
}
_WORD_OPS = {
    "forall": "∀", "exists": "∃", "and": "∧", "or": "∨", "not": "¬",
    "implies": "→", "iff": "↔", "xor": "⊕",
}
_FREE_VARIABLE_NAMES = re.compile(r"^[a-z]\d*$")


class FOLParseError(ValueError):
    pass


def _tokenize(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN_RE.finditer(text):
        if match.group("op"):
            tok = match.group("op")
            tokens.append(_OP_ALIASES.get(tok, tok))
        elif match.group("id"):
            tok = match.group("id").rstrip(".-")
            if tok.lower() in _WORD_OPS and tok.lower() == tok:
                tokens.append(_WORD_OPS[tok])
            elif tok == "v" and tokens and tokens[-1] == ")":
                tokens.append("∨")  # "P(a) v Q(a)": a term cannot follow ")"
            elif tok:
                tokens.append(tok)
        # Unknown junk characters (stray quotes, em-dashes) are ignored.
    return tokens


def _balance(tokens: list[str]) -> list[str]:
    """Repair unbalanced parentheses, a frequent annotation error in FOLIO."""
    depth = 0
    out = []
    for tok in tokens:
        if tok == "(":
            depth += 1
        elif tok == ")":
            if depth == 0:
                continue
            depth -= 1
        out.append(tok)
    return out + [")"] * depth


class _FOLParser:
    # Precedence (loosest first): ↔, →, ⊕, ∨, ∧, ¬/quantifiers.
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.scope: list[dict[str, Var]] = []

    def peek(self, offset=0):
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def take(self, expected=None):
        tok = self.peek()
        if tok is None or (expected is not None and tok != expected):
            raise FOLParseError(f"expected {expected!r}, got {tok!r}")
        self.pos += 1
        return tok

    def parse(self):
        f = self.iff()
        if self.peek() is not None:
            raise FOLParseError(f"trailing tokens: {self.tokens[self.pos:]}")
        return f

    def iff(self):
        left = self.imp()
        while self.peek() == "↔":
            self.take()
            left = ("iff", left, self.imp())
        return left

    def imp(self):
        left = self.xor()
        if self.peek() == "→":
            self.take()
            return ("imp", left, self.imp())
        return left

    def xor(self):
        left = self.disj()
        while self.peek() == "⊕":
            self.take()
            left = ("xor", left, self.disj())
        return left

    def disj(self):
        parts = [self.conj()]
        while self.peek() == "∨":
            self.take()
            parts.append(self.conj())
        return parts[0] if len(parts) == 1 else ("or", *parts)

    def conj(self):
        parts = [self.unary()]
        while self.peek() == "∧":
            self.take()
            parts.append(self.unary())
        return parts[0] if len(parts) == 1 else ("and", *parts)

    def unary(self):
        tok = self.peek()
        if tok == "¬":
            self.take()
            return ("not", self.unary())
        if tok in ("∀", "∃"):
            self.take()
            names = []
            while self.peek() not in (None, "(", "¬", "∀", "∃") and self._is_name(
                self.peek()
            ):
                # "∀x ∀y" or "∀x y" or "∀x,y"
                names.append(self.take())
                if self.peek() == ",":
                    self.take()
                if self.peek(1) == "(" and self.peek() not in ("∀", "∃"):
                    break
            op = "all" if tok == "∀" else "ex"
            if not names:
                # "∃(P(x) ∧ Q(x))": bind the free variables of the scope.
                if self.peek() != "(":
                    raise FOLParseError("quantifier without variable")
                free: dict[str, Var] = {}
                body = _bind_free(self.iff(), free)
                for v in reversed(free.values()):
                    body = (op, v, body)
                return body
            variables = [Var(n) for n in names]
            self.scope.append(dict(zip(names, variables)))
            # Quantifier scope extends as far right as possible.
            body = self.iff()
            self.scope.pop()
            for v in reversed(variables):
                body = (op, v, body)
            return body
        if tok == "(":
            self.take()
            f = self.iff()
            if self.peek() == ")":
                self.take()
            return f
        return self.atomic()

    @staticmethod
    def _is_name(tok):
        return tok is not None and re.match(r"^\w", tok) is not None

    def lookup(self, name):
        for frame in reversed(self.scope):
            if name in frame:
                return frame[name]
        return None

    def term(self):
        name = self.take()
        if not self._is_name(name):
            raise FOLParseError(f"bad term {name!r}")
        if self.peek() == "(":
            self.take()
            args = [self.term()]
            while self.peek() == ",":
                self.take()
                args.append(self.term())
            if self.peek() == ")":
                self.take()
            return (name, *args)
        bound = self.lookup(name)
        if bound is not None:
            return bound
        if _FREE_VARIABLE_NAMES.match(name):
            return ("$free", name)
        return name

    def atomic(self):
        tok = self.peek()
        if not self._is_name(tok):
            raise FOLParseError(f"unexpected token {tok!r}")
        if self.peek(1) == "(":
            name = self.take()
            self.take("(")
            args = []
            if self.peek() != ")":
                args.append(self.term())
                while self.peek() == ",":
                    self.take()
                    args.append(self.term())
            if self.peek() == ")":
                self.take()
            atom = ("atom", name, tuple(args))
        else:
            left = self.term()
            if self.peek() in ("=", "≠"):
                op = self.take()
                right = self.term()
                atom = ("atom", "=", (left, right))
                return atom if op == "=" else ("not", atom)
            if type(left) is tuple and left[0] == "$free":
                left = left[1]
            atom = ("atom", str(left), ())
        if self.peek() in ("=", "≠"):
            # A predicate application used as a term: P(x) = c (rare).
            op = self.take()
            right = self.term()
            eq = ("atom", "=", ((atom[1], *atom[2]), right))
            return eq if op == "=" else ("not", eq)
        return atom


def _bind_free(f, free: dict[str, Var]):
    """Replace placeholder free variables by shared universally bound Vars."""
    op = f[0]
    if op == "atom":
        return ("atom", f[1], tuple(_bind_term(t, free) for t in f[2]))
    if op in ("all", "ex"):
        return (op, f[1], _bind_free(f[2], free))
    if op in ("true", "false"):
        return f
    return (op, *(_bind_free(g, free) for g in f[1:]))


def _bind_term(t, free):
    if type(t) is tuple:
        if t[0] == "$free":
            if t[1] not in free:
                free[t[1]] = Var(t[1])
            return free[t[1]]
        return (t[0], *(_bind_term(a, free) for a in t[1:]))
    return t


def parse_fol(text: str, free_quantifier: str = "all"):
    """Parse one FOL formula.

    Free variables are closed with ``free_quantifier``: universally for
    premises, existentially for goals ("Cost(gre, x) ∧ Cheaper(x, 300)").
    """
    tokens = _balance(_tokenize(text))
    if not tokens:
        raise FOLParseError("empty formula")
    f = _FOLParser(tokens).parse()
    free: dict[str, Var] = {}
    f = _bind_free(f, free)
    for v in free.values():
        f = (free_quantifier, v, f)
    return f


# ---------------------------------------------------------------------------
# Clause normal form
# ---------------------------------------------------------------------------


def _nnf(f, positive=True):
    """Negation normal form with →, ↔, ⊕ eliminated."""
    op = f[0]
    if op == "atom":
        return f if positive else ("not", f)
    if op == "true":
        return f if positive else ("false",)
    if op == "false":
        return f if positive else ("true",)
    if op == "not":
        return _nnf(f[1], not positive)
    if op == "and":
        parts = [_nnf(g, positive) for g in f[1:]]
        return ("and", *parts) if positive else ("or", *parts)
    if op == "or":
        parts = [_nnf(g, positive) for g in f[1:]]
        return ("or", *parts) if positive else ("and", *parts)
    if op == "imp":
        return _nnf(("or", ("not", f[1]), f[2]), positive)
    if op == "iff":
        a, b = f[1], f[2]
        if positive:
            return _nnf(("and", ("imp", a, b), ("imp", b, a)))
        return _nnf(("xor", a, b))
    if op == "xor":
        a, b = f[1], f[2]
        if positive:
            return ("and", ("or", _nnf(a), _nnf(b)), ("or", _nnf(a, False), _nnf(b, False)))
        return _nnf(("iff", a, b))
    if op in ("all", "ex"):
        flip = {"all": "ex", "ex": "all"}
        return ((op if positive else flip[op]), f[1], _nnf(f[2], positive))
    raise ValueError(f"unknown operator {op}")


class _Skolemizer:
    def __init__(self, prefix="sk"):
        self.counter = itertools.count()
        self.prefix = prefix

    def run(self, f, env: dict, universals: tuple):
        op = f[0]
        if op == "atom":
            return ("atom", f[1], tuple(_subst_term(t, env) for t in f[2]))
        if op == "not":
            return ("not", self.run(f[1], env, universals))
        if op in ("and", "or"):
            return (op, *(self.run(g, env, universals) for g in f[1:]))
        if op == "all":
            fresh = Var(f[1].name)
            return self.run(f[2], {**env, f[1]: fresh}, universals + (fresh,))
        if op == "ex":
            name = f"{self.prefix}{next(self.counter)}"
            sk = (name, *universals) if universals else name
            return self.run(f[2], {**env, f[1]: sk}, universals)
        return f


def _subst_term(t, env):
    if type(t) is Var:
        return env.get(t, t)
    if type(t) is tuple:
        return (t[0], *(_subst_term(a, env) for a in t[1:]))
    return t


def _to_literal(f):
    if f[0] == "not":
        return (False, f[1][1], f[1][2])
    return (True, f[1], f[2])


_definition_counter = itertools.count()


def _cnf_clauses(f, limit: int, extra: list) -> list[frozenset]:
    """Clauses for an NNF, quantifier-free formula.

    Distribution is used while the result stays small; larger disjuncts are
    replaced by fresh definition atoms (Tseitin), whose defining clauses are
    appended to ``extra``. This preserves satisfiability.
    """
    op = f[0]
    if op in ("atom", "not"):
        return [frozenset([_to_literal(f)])]
    if op == "true":
        return []
    if op == "false":
        return [frozenset()]
    if op == "and":
        out = []
        for g in f[1:]:
            out.extend(_cnf_clauses(g, limit, extra))
        return out
    if op == "or":
        subs = [_cnf_clauses(g, limit, extra) for g in f[1:]]
        size = 1
        for sub in subs:
            size *= max(1, len(sub))
        if size > limit:
            renamed = []
            for g, sub in zip(f[1:], subs):
                if len(sub) > 1:
                    args = tuple(sorted(free_vars_of_formula(g), key=lambda v: v.id))
                    name = f"def{next(_definition_counter)}"
                    for c in sub:
                        extra.append(c | {(False, name, args)})
                    sub = [frozenset([(True, name, args)])]
                renamed.append(sub)
            subs = renamed
        result = [frozenset()]
        for sub in subs:
            result = [a | b for a in result for b in sub]
        return result
    raise ValueError(op)


def _is_tautology(clause) -> bool:
    return any((not p, q, a) in clause for (p, q, a) in clause)


def to_clauses(f, skolemizer: _Skolemizer | None = None, limit: int = 64):
    """Convert a closed formula into a list of clauses (tuples of literals)."""
    skolemizer = skolemizer or _Skolemizer()
    g = skolemizer.run(_nnf(f), {}, ())
    extra: list = []
    clauses = _cnf_clauses(g, limit, extra) + extra
    out = []
    seen = set()
    for c in clauses:
        if _is_tautology(c):
            continue
        key = tuple(sorted(c, key=repr))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def free_vars_of_formula(f, bound=frozenset()):
    op = f[0]
    if op == "atom":
        out = set()
        for t in f[2]:
            out |= _term_vars(t)
        return out - bound
    if op in ("all", "ex"):
        return free_vars_of_formula(f[2], bound | {f[1]})
    if op in ("true", "false"):
        return set()
    out = set()
    for g in f[1:]:
        out |= free_vars_of_formula(g, bound)
    return out


def _term_vars(t):
    if type(t) is Var:
        return {t}
    if type(t) is tuple:
        out = set()
        for a in t[1:]:
            out |= _term_vars(a)
        return out
    return set()


def close_universally(f):
    for v in free_vars_of_formula(f):
        f = ("all", v, f)
    return f


def _map_formula(f, pred_map, const_map):
    op = f[0]
    if op == "atom":
        return ("atom", pred_map.get((f[1], len(f[2])), f[1]),
                tuple(_map_term(t, const_map) for t in f[2]))
    if op in ("all", "ex"):
        return (op, f[1], _map_formula(f[2], pred_map, const_map))
    if op in ("true", "false"):
        return f
    return (op, *(_map_formula(g, pred_map, const_map) for g in f[1:]))


def _map_term(t, const_map):
    if type(t) is tuple:
        return (t[0], *(_map_term(a, const_map) for a in t[1:]))
    if type(t) is str:
        return const_map.get(t, t)
    return t


def _symbol_counts(f, preds, consts):
    op = f[0]
    if op == "atom":
        preds[(f[1], len(f[2]))] = preds.get((f[1], len(f[2])), 0) + 1
        for t in f[2]:
            _const_counts(t, consts)
    elif op in ("all", "ex"):
        _symbol_counts(f[2], preds, consts)
    elif op not in ("true", "false"):
        for g in f[1:]:
            _symbol_counts(g, preds, consts)


def _const_counts(t, consts):
    if type(t) is tuple:
        for a in t[1:]:
            _const_counts(a, consts)
    elif type(t) is str:
        consts[t] = consts.get(t, 0) + 1


def _norm_symbol(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _similar(a: str, b: str) -> bool:
    import difflib

    x, y = _norm_symbol(a), _norm_symbol(b)
    if not x or not y:
        return False
    if x == y:
        return True
    shorter, longer = sorted((x, y), key=len)
    if len(shorter) >= 4 and longer.startswith(shorter) and len(longer) - len(shorter) <= 3:
        return True
    if len(shorter) >= 4 and longer.endswith(shorter) and len(longer) - len(shorter) <= 1:
        return True
    return len(shorter) >= 5 and difflib.SequenceMatcher(None, x, y).ratio() >= 0.86


def repair_symbols(premises, goal):
    """Merge near-duplicate symbols (typos / inconsistent naming).

    A symbol that occurs only once across the problem is renamed to a
    sufficiently similar symbol of the same kind and arity that occurs more
    often; a symbol used only by the goal is renamed to a similar premise
    symbol. Exact case-insensitive matches are always merged.
    """
    preds, consts = {}, {}
    for f in premises:
        _symbol_counts(f, preds, consts)
    premise_syms = set(preds) | set(consts)
    _symbol_counts(goal, preds, consts)

    def build(counts, key_name, key_arity):
        mapping = {}
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        for sym, n in items:
            name, arity = key_name(sym), key_arity(sym)
            for other, m in items:
                if other == sym or key_arity(other) != arity or other in mapping:
                    continue
                oname = key_name(other)
                exact = _norm_symbol(oname) == _norm_symbol(name)
                if exact and (m > n or (m == n and oname < name)):
                    mapping[sym] = oname
                    break
                goal_only = sym not in premise_syms and other in premise_syms
                if (goal_only or (n == 1 and m >= 2)) and _similar(name, oname):
                    mapping[sym] = oname
                    break
        return mapping

    pred_map = build(preds, lambda s: s[0], lambda s: s[1])
    const_map = build(consts, lambda s: s, lambda s: 0)
    const_map = {k: v for k, v in const_map.items() if not k.isdigit() and not v.isdigit()}
    if not pred_map and not const_map:
        return list(premises), goal, {}
    fixed = [_map_formula(f, pred_map, const_map) for f in premises]
    changes = {**{f"{k[0]}/{k[1]}": v for k, v in pred_map.items()}, **const_map}
    return fixed, _map_formula(goal, pred_map, const_map), changes


def predicates_of(f, out=None):
    out = set() if out is None else out
    op = f[0]
    if op == "atom":
        out.add((f[1], len(f[2])))
    elif op in ("all", "ex"):
        predicates_of(f[2], out)
    elif op not in ("true", "false"):
        for g in f[1:]:
            predicates_of(g, out)
    return out
