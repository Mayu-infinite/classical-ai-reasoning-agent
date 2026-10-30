"""Rule engine for function-free Horn rules with negation.

A rule is ``body -> head`` where body literals may be negated and the head is a
single literal (positive, or negative = "strong"/explicit negation).

Two semantics are provided:

* ``cwa`` (closed world): a negated body literal is *negation as failure*.
  Models are computed with the well-founded semantics by the alternating
  fixpoint (Van Gelder), which equals the perfect model for stratified
  programs. Each inner step is forward chaining to a least fixpoint.
* ``owa`` (open world): negated literals are ordinary facts about a
  complementary predicate, so ``¬p`` is only true if it is stated/derived.

Literals are ``(positive, pred, args)``; variables are ``Var`` objects.
"""

from dataclasses import dataclass, field

from reasoning_agent.logic import Var, atom_str, literal_str


@dataclass(frozen=True)
class Rule:
    body: tuple  # tuple of literals
    head: tuple  # a single literal
    text: str = ""

    def __str__(self):
        names: dict[Var, str] = {}

        def show(lit):
            positive, pred, args = lit
            shown = []
            for a in args:
                if type(a) is Var:
                    if a not in names:
                        i = len(names)
                        names[a] = "XYZWUV"[i] if i < 6 else f"X{i}"
                    a = names[a]
                shown.append(a)
            return literal_str((positive, pred, tuple(shown)))

        body = [show(b) for b in self.body]
        head = show(self.head)
        return " ∧ ".join(body) + " → " + head if body else head


def _ground(args, binding):
    return tuple(binding.get(a, a) if type(a) is Var else a for a in args)


def _match_args(pattern, fact_args, binding):
    for p, f in zip(pattern, fact_args):
        if type(p) is Var:
            bound = binding.get(p)
            if bound is None:
                binding = {**binding, p: f}
            elif bound != f:
                return None
        elif p != f:
            return None
    return binding


@dataclass
class Model:
    true: set  # set of (pred, args) atoms that are true
    possible: set  # superset: atoms not known to be false
    neg: set = field(default_factory=set)  # explicitly derived ¬atoms
    why: dict = field(default_factory=dict)  # literal -> (rule, body instances)
    iterations: int = 0
    rule_firings: int = 0

    def holds(self, positive, pred, args, cwa=True):
        atom = (pred, args)
        if positive:
            return atom in self.true
        if (pred, args) in self.neg:
            return True
        return cwa and atom not in self.possible


class ForwardChainer:
    def __init__(self, rules, domain=(), cwa=True, max_facts=200_000):
        self.rules = list(rules)
        self.cwa = cwa
        self.max_facts = max_facts
        consts = set(domain)
        for r in self.rules:
            for _, _, args in (*r.body, r.head):
                consts.update(a for a in args if type(a) is not Var)
        self.domain = sorted(consts)
        self.firings = 0

    # -- core least-fixpoint computation --------------------------------------

    def _least_model(self, naf_reference, record=None):
        """Least fixpoint, NAF literals evaluated against ``naf_reference``.

        ``naf_reference`` is a set of atoms; ``not a`` holds iff a ∉ it.
        In OWA mode negative literals are separate "¬" predicates instead.
        """
        facts: dict[tuple, set] = {}  # (positive, pred) -> set(args)
        new_by_key: dict[tuple, set] = {}

        def add(lit, rule, body):
            key = (lit[0], lit[1])
            bucket = facts.setdefault(key, set())
            if lit[2] in bucket:
                return False
            bucket.add(lit[2])
            new_by_key.setdefault(key, set()).add(lit[2])
            if record is not None and (lit not in record):
                record[lit] = (rule, body)
            return True

        pending = []
        for r in self.rules:
            if not r.body:
                head = r.head
                if any(type(a) is Var for a in head[2]):
                    for c in self.domain:
                        g = tuple(c if type(a) is Var else a for a in head[2])
                        add((head[0], head[1], g), r, ())
                else:
                    add(head, r, ())
            else:
                pending.append(r)

        split = []
        for r in pending:
            pos = [b for b in r.body if b[0] or not self.cwa]
            neg = [b for b in r.body if not b[0] and self.cwa]
            pos.sort(key=lambda b: -len(b[2]))
            split.append((r, pos, neg))

        first = True
        while True:
            changed = False
            delta = new_by_key
            new_by_key = {}
            for r, pos, neg in split:
                # Semi-naive: some positive literal must match a fact that was
                # new in the previous round; rules without positive literals
                # only need to be tried in the first round.
                for binding in self._join(pos, facts, delta, first):
                    for full in self._bindings_for_neg(neg, binding):
                        ok = True
                        for b in neg:
                            atom = (b[1], _ground(b[2], full))
                            if atom in naf_reference:
                                ok = False
                                break
                        if not ok:
                            continue
                        head_args = _ground(r.head[2], full)
                        if any(type(a) is Var for a in head_args):
                            heads = [
                                tuple(c if type(a) is Var else a for a in head_args)
                                for c in self.domain
                            ]
                        else:
                            heads = [head_args]
                        for h in heads:
                            self.firings += 1
                            body_inst = tuple(
                                (b[0], b[1], _ground(b[2], full)) for b in r.body
                            )
                            if add((r.head[0], r.head[1], h), r, body_inst):
                                changed = True
                if sum(len(v) for v in facts.values()) > self.max_facts:
                    raise RuntimeError("fact limit exceeded")
            first = False
            if not changed:
                break
        return facts

    def _bindings_for_neg(self, neg, binding):
        unbound = []
        for b in neg:
            for a in b[2]:
                if type(a) is Var and a not in binding and a not in unbound:
                    unbound.append(a)
        if not unbound:
            yield binding
            return

        def rec(i, bnd):
            if i == len(unbound):
                yield bnd
                return
            for c in self.domain:
                yield from rec(i + 1, {**bnd, unbound[i]: c})

        yield from rec(0, binding)

    def _join(self, pos, facts, delta, first):
        """Yield bindings satisfying all positive literals.

        Semi-naive evaluation: at least one literal must match a fact from
        ``delta`` (facts new in the previous round). On the first round,
        ``delta`` contains every fact.
        """
        if not pos:
            if first:
                yield {}
            return
        for pivot in range(len(pos)):
            key = (pos[pivot][0], pos[pivot][1])
            if not delta.get(key):
                continue
            yield from self._join_rec(pos, 0, {}, facts, delta, pivot)

    def _join_rec(self, pos, i, binding, facts, delta, pivot):
        if i == len(pos):
            yield binding
            return
        lit = pos[i]
        key = (lit[0], lit[1])
        if i == pivot:
            source = delta.get(key, ())
        elif i < pivot:
            # Literals before the pivot use old facts only (avoid duplicates).
            source = facts.get(key, set()) - delta.get(key, set())
        else:
            source = facts.get(key, ())
        pattern = lit[2]
        if all(type(a) is not Var or a in binding for a in pattern):
            g = _ground(pattern, binding)
            if g in source:
                yield from self._join_rec(pos, i + 1, binding, facts, delta, pivot)
            return
        for fact_args in list(source):
            if len(fact_args) != len(pattern):
                continue
            b2 = _match_args(pattern, fact_args, binding)
            if b2 is not None:
                yield from self._join_rec(pos, i + 1, b2, facts, delta, pivot)

    # -- semantics --------------------------------------------------------------

    def model(self, max_rounds=50) -> Model:
        if not self.cwa:
            record = {}
            facts = self._least_model(set(), record)
            true = {(p, a) for (s, p), args in facts.items() if s for a in args}
            neg = {(p, a) for (s, p), args in facts.items() if not s for a in args}
            return Model(true, true | neg, neg, record, 1, self.firings)

        # Alternating fixpoint for the well-founded semantics.
        has_naf = any(not b[0] for r in self.rules for b in r.body)
        everything = _Universe()
        under_ref = everything  # NAF false everywhere -> under-approximation
        rounds = 0
        prev_true = None
        true_atoms = set()
        possible = set()
        record = {}
        while rounds < max_rounds:
            rounds += 1
            rec = {}
            facts_t = self._least_model(under_ref if rounds > 1 else everything, rec)
            true_atoms = {(p, a) for (s, p), args in facts_t.items() if s for a in args}
            neg_atoms = {(p, a) for (s, p), args in facts_t.items() if not s for a in args}
            record = rec
            if not has_naf:
                possible = true_atoms
                break
            facts_u = self._least_model(true_atoms)
            possible = {(p, a) for (s, p), args in facts_u.items() if s for a in args}
            if true_atoms == prev_true:
                break
            prev_true = true_atoms
            under_ref = possible
        return Model(true_atoms, possible, neg_atoms, record, rounds, self.firings)


class _Universe:
    """A set-like object containing every atom (NAF fails everywhere)."""

    def __contains__(self, item):
        return True


def explain(model: Model, lit, depth=0, seen=None, max_depth=25) -> list[str]:
    """Human-readable proof tree for a derived literal."""
    seen = set() if seen is None else seen
    pad = "  " * depth
    positive, pred, args = lit
    key = (positive, pred, args)
    entry = model.why.get(key)
    text = literal_str(lit)
    if entry is None:
        if not positive:
            return [f"{pad}{text}   [negation as failure: {atom_str(pred, args)} is not derivable]"]
        return [f"{pad}{text}   [not derivable]"]
    rule, body = entry
    if not body:
        return [f"{pad}{text}   [fact]"]
    if key in seen or depth > max_depth:
        return [f"{pad}{text}   [see above]"]
    seen.add(key)
    lines = [f"{pad}{text}   [rule: {rule}]"]
    for b in body:
        lines.extend(explain(model, b, depth + 1, seen, max_depth))
    return lines


# ---------------------------------------------------------------------------
# Backward chaining (SLDNF with loop checking), an alternative strategy.
# ---------------------------------------------------------------------------


class BackwardChainer:
    """Goal-directed proof search over the same rules.

    Negated goals use negation as failure (CWA) or explicit negation (OWA).
    Ground goals are tabled to avoid repeated work and infinite loops.
    """

    def __init__(self, rules, domain=(), cwa=True, max_depth=60):
        self.cwa = cwa
        self.max_depth = max_depth
        self.by_head: dict[tuple, list[Rule]] = {}
        consts = set(domain)
        for r in rules:
            self.by_head.setdefault((r.head[0], r.head[1]), []).append(r)
            for _, _, args in (*r.body, r.head):
                consts.update(a for a in args if type(a) is not Var)
        self.domain = sorted(consts)
        self.table: dict[tuple, bool] = {}
        self.calls = 0

    def prove(self, lit) -> bool:
        return self._prove(lit, frozenset(), 0)

    def _prove(self, lit, stack, depth) -> bool:
        positive, pred, args = lit
        if not positive and self.cwa:
            # Negation as failure. In a stratified program the negated goal
            # does not depend on its ancestors, so it starts a fresh loop check.
            return not self._prove_atom((True, pred, args), frozenset(), depth)
        return self._prove_atom(lit, stack, depth)

    def _prove_atom(self, lit, stack, depth) -> bool:
        self.calls += 1
        if lit in self.table:
            return self.table[lit]
        if lit in stack or depth > self.max_depth:
            return False
        stack = stack | {lit}
        result = False
        for r in self.by_head.get((lit[0], lit[1]), ()):
            binding = _match_args(r.head[2], lit[2], {})
            if binding is None:
                continue
            if self._prove_body(list(r.body), binding, stack, depth + 1):
                result = True
                break
        if result or not stack - {lit}:
            self.table[lit] = result
        return result

    def _prove_body(self, body, binding, stack, depth) -> bool:
        if not body:
            return True
        # Prefer a literal whose arguments are already ground.
        body.sort(key=lambda b: (not b[0], sum(type(a) is Var and a not in binding for a in b[2])))
        lit, rest = body[0], body[1:]
        args = _ground(lit[2], binding)
        unbound = [a for a in args if type(a) is Var]
        if not unbound:
            return self._prove((lit[0], lit[1], args), stack, depth) and self._prove_body(
                rest, binding, stack, depth
            )
        for values in _product(self.domain, len(set(unbound))):
            b2 = dict(binding)
            for v, c in zip(dict.fromkeys(unbound), values):
                b2[v] = c
            g = _ground(lit[2], b2)
            if self._prove((lit[0], lit[1], g), stack, depth) and self._prove_body(
                list(rest), b2, stack, depth
            ):
                return True
        return False


def _product(domain, n):
    if n == 0:
        yield ()
        return
    for c in domain:
        for rest in _product(domain, n - 1):
            yield (c, *rest)
