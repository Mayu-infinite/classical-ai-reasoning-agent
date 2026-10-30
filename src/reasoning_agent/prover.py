"""Refutation theorem prover for first-order clauses.

Algorithm: binary resolution + factoring inside a given-clause loop.

* Set-of-support: only clauses descending from the negated goal are selected
  as given clauses; the knowledge-base clauses act as the "usable" side.
  This keeps the search goal-directed and avoids "proving" queries from an
  inconsistent annotation.
* Clause selection is best-first (uniform-cost style) on clause weight
  (number of symbols), interleaved with oldest-first picks for fairness.
* Forward subsumption and tautology deletion prune the search space.
* Equality, when present, is axiomatised (reflexivity, symmetry,
  transitivity, and predicate substitutivity).
"""

import heapq
import time
from dataclasses import dataclass, field

from reasoning_agent.logic import Var, clause_str, to_clauses

# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------


def walk(t, s):
    while type(t) is Var and t in s:
        t = s[t]
    return t


def occurs(v, t, s) -> bool:
    t = walk(t, s)
    if t is v:
        return True
    if type(t) is tuple:
        return any(occurs(v, a, s) for a in t[1:])
    return False


def unify(a, b, s):
    """Most general unifier extending substitution ``s`` (or None)."""
    a = walk(a, s)
    b = walk(b, s)
    if a is b:
        return s
    if type(a) is Var:
        if occurs(a, b, s):
            return None
        s = dict(s)
        s[a] = b
        return s
    if type(b) is Var:
        if occurs(b, a, s):
            return None
        s = dict(s)
        s[b] = a
        return s
    if type(a) is tuple and type(b) is tuple:
        if len(a) != len(b) or a[0] != b[0]:
            return None
        for x, y in zip(a[1:], b[1:]):
            s = unify(x, y, s)
            if s is None:
                return None
        return s
    return s if a == b else None


def unify_args(xs, ys, s):
    if len(xs) != len(ys):
        return None
    for x, y in zip(xs, ys):
        s = unify(x, y, s)
        if s is None:
            return None
    return s


def substitute(t, s):
    t = walk(t, s)
    if type(t) is tuple:
        return (t[0], *(substitute(a, s) for a in t[1:]))
    return t


def match(pattern, target, s):
    """One-way matching: bind only variables of ``pattern``."""
    if type(pattern) is Var:
        if pattern in s:
            return s if s[pattern] == target else None
        s = dict(s)
        s[pattern] = target
        return s
    if type(pattern) is tuple:
        if type(target) is not tuple or len(pattern) != len(target):
            return None
        if pattern[0] != target[0]:
            return None
        for x, y in zip(pattern[1:], target[1:]):
            s = match(x, y, s)
            if s is None:
                return None
        return s
    return s if pattern == target else None


# ---------------------------------------------------------------------------
# Clause utilities
# ---------------------------------------------------------------------------


def clause_vars(clause):
    out = []
    seen = set()

    def visit(t):
        if type(t) is Var:
            if t not in seen:
                seen.add(t)
                out.append(t)
        elif type(t) is tuple:
            for a in t[1:]:
                visit(a)

    for _, _, args in clause:
        for a in args:
            visit(a)
    return out


def rename(clause):
    vs = clause_vars(clause)
    if not vs:
        return clause
    s = {v: Var(v.name) for v in vs}
    return tuple((p, q, tuple(substitute(a, s) for a in args)) for p, q, args in clause)


def apply_clause(lits, s):
    out = []
    for p, q, args in lits:
        lit = (p, q, tuple(substitute(a, s) for a in args))
        if lit not in out:
            out.append(lit)
    return tuple(out)


def is_tautology(clause) -> bool:
    for p, q, a in clause:
        if p and (False, q, a) in clause:
            return True
        if p and q == "=" and len(a) == 2 and a[0] == a[1]:
            return True
    return False


def _term_weight(t):
    if type(t) is tuple:
        return 1 + sum(_term_weight(a) for a in t[1:])
    return 1


def weight(clause) -> int:
    return sum(1 + sum(_term_weight(a) for a in args) for _, _, args in clause)


def canonical(clause):
    """Variant-invariant key (variables numbered by first occurrence)."""
    mapping = {}

    def canon(t):
        if type(t) is Var:
            if t not in mapping:
                mapping[t] = f"?{len(mapping)}"
            return mapping[t]
        if type(t) is tuple:
            return (t[0], *(canon(a) for a in t[1:]))
        return t

    lits = sorted(clause, key=lambda lit: (lit[1], lit[0], repr(lit[2])))
    return tuple((p, q, tuple(canon(a) for a in args)) for p, q, args in lits)


def subsumes(c, d) -> bool:
    """True if clause c θ-subsumes clause d."""
    if len(c) > len(d):
        return False

    def search(i, s):
        if i == len(c):
            return True
        p, q, args = c[i]
        for p2, q2, args2 in d:
            if p2 == p and q2 == q and len(args2) == len(args):
                s2 = s
                for x, y in zip(args, args2):
                    s2 = match(x, y, s2)
                    if s2 is None:
                        break
                if s2 is not None and search(i + 1, s2):
                    return True
        return False

    return search(0, {})


# ---------------------------------------------------------------------------
# Prover
# ---------------------------------------------------------------------------


@dataclass
class ProofResult:
    proved: bool
    reason: str
    steps: int = 0
    clauses: int = 0
    proof: list[str] = field(default_factory=list)


def equality_axioms(clauses):
    preds = {}
    for c in clauses:
        for _, q, args in c:
            if q != "=":
                preds[q] = len(args)
    x, y, z = Var("x"), Var("y"), Var("z")
    axioms = [
        ((True, "=", (x, x)),),
        ((False, "=", (x, y)), (True, "=", (y, x))),
        ((False, "=", (x, y)), (False, "=", (y, z)), (True, "=", (x, z))),
    ]
    for q, n in preds.items():
        for i in range(n):
            a = [Var("a") for _ in range(n)]
            b = list(a)
            u, v = Var("u"), Var("v")
            a[i], b[i] = u, v
            axioms.append(((False, "=", (u, v)), (False, q, tuple(a)), (True, q, tuple(b))))
    return axioms


def _depth(t):
    if type(t) is tuple:
        return 1 + max((_depth(a) for a in t[1:]), default=0)
    return 0


def remove_pure(axioms, sos):
    """Drop clauses containing a literal whose complement occurs nowhere.

    Such clauses can never take part in a refutation (pure literal rule).
    """
    while True:
        signs = {(p, q) for c in axioms + sos for p, q, _ in c}
        keep = lambda c: all((not p, q) in signs for p, q, _ in c)  # noqa: E731
        new_axioms = [c for c in axioms if keep(c)]
        new_sos = [c for c in sos if keep(c)]
        if len(new_axioms) == len(axioms) and len(new_sos) == len(sos):
            return axioms, sos
        axioms, sos = new_axioms, new_sos


class Prover:
    def __init__(self, max_steps=3000, max_clauses=30000, time_limit=2.0, age_ratio=5,
                 max_literals=8, max_depth=3):
        self.max_steps = max_steps
        self.max_clauses = max_clauses
        self.time_limit = time_limit
        self.age_ratio = age_ratio
        self.max_literals = max_literals
        self.max_depth = max_depth

    def refute(self, axioms, sos) -> ProofResult:
        """Search for the empty clause from ``axioms`` ∪ ``sos``."""
        deadline = time.perf_counter() + self.time_limit
        axioms = [c for c in axioms if not is_tautology(c)]
        sos = [c for c in sos if not is_tautology(c)]
        if any(q == "=" for c in axioms + sos for _, q, _ in c):
            axioms = axioms + equality_axioms(axioms + sos)
        axioms, sos = remove_pure(axioms, sos)
        if not sos:
            return ProofResult(False, "search exhausted")

        store: list[tuple] = []  # clause id -> clause
        parents: list = []
        seen = set()
        usable_by_lit: dict[tuple, list[int]] = {}
        usable_ids: list[int] = []
        weight_heap: list = []
        age_heap: list = []
        selected = set()

        def add(clause, origin):
            key = canonical(clause)
            if key in seen:
                return None
            seen.add(key)
            cid = len(store)
            store.append(clause)
            parents.append(origin)
            return cid

        subsume_index: dict[tuple, list[int]] = {}

        def make_usable(cid):
            usable_ids.append(cid)
            clause = store[cid]
            for i, (p, q, _) in enumerate(clause):
                usable_by_lit.setdefault((p, q), []).append((cid, i))
            if clause:
                # A subsumer must share all its literal keys with the subsumed
                # clause, so indexing it under one key finds it exactly once.
                p, q, _ = clause[0]
                subsume_index.setdefault((p, q), []).append(cid)

        for c in axioms:
            cid = add(c, ("axiom",))
            if cid is not None:
                make_usable(cid)

        def push(cid):
            w = weight(store[cid])
            heapq.heappush(weight_heap, (w, cid))
            heapq.heappush(age_heap, (cid, cid))

        for c in sos:
            if not c:
                return ProofResult(True, "empty clause in input", proof=["□"])
            cid = add(c, ("goal",))
            if cid is not None:
                push(cid)

        def forward_subsumed(clause):
            preds = {(p, q) for p, q, _ in clause}
            n = len(clause)
            for key in preds:
                for cid in subsume_index.get(key, ()):
                    other = store[cid]
                    if len(other) <= n and all((a, b) in preds for a, b, _ in other):
                        if subsumes(other, clause):
                            return True
            return False

        steps = 0
        while weight_heap or age_heap:
            if steps >= self.max_steps or len(store) >= self.max_clauses:
                return ProofResult(False, "resource limit", steps, len(store))
            if time.perf_counter() > deadline:
                return ProofResult(False, "time limit", steps, len(store))
            heap = age_heap if (steps % self.age_ratio == self.age_ratio - 1) else weight_heap
            if not heap:
                heap = weight_heap or age_heap
            _, gid = heapq.heappop(heap)
            if gid in selected:
                continue
            selected.add(gid)
            given = store[gid]
            steps += 1
            if forward_subsumed(given):
                continue
            make_usable(gid)
            g = rename(given)

            new = []
            # Binary resolution against every usable clause (including itself).
            for i, (p, q, args) in enumerate(g):
                for oid, j in list(usable_by_lit.get((not p, q), ())):
                    other = rename(store[oid]) if oid != gid else rename(g)
                    _, _, oargs = other[j]
                    s = unify_args(args, oargs, {})
                    if s is None:
                        continue
                    lits = g[:i] + g[i + 1 :] + other[:j] + other[j + 1 :]
                    new.append((apply_clause(lits, s), ("res", gid, oid)))
            # Factoring of the given clause.
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    if g[i][0] == g[j][0] and g[i][1] == g[j][1]:
                        s = unify_args(g[i][2], g[j][2], {})
                        if s is not None:
                            new.append((apply_clause(g[:j] + g[j + 1 :], s), ("fac", gid)))

            for clause, origin in new:
                if not clause:
                    cid = len(store)
                    store.append(clause)
                    parents.append(origin)
                    proof = self._trace(store, parents, cid)
                    return ProofResult(True, "refutation found", steps, len(store), proof)
                if len(clause) > self.max_literals or any(
                    _depth(a) > self.max_depth for _, _, args in clause for a in args
                ):
                    continue
                if is_tautology(clause) or forward_subsumed(clause):
                    continue
                cid = add(clause, origin)
                if cid is not None:
                    push(cid)
        return ProofResult(False, "search exhausted", steps, len(store))

    @staticmethod
    def _trace(store, parents, cid) -> list[str]:
        order = []
        visited = set()

        def visit(i):
            if i in visited:
                return
            visited.add(i)
            origin = parents[i]
            for p in origin[1:]:
                visit(p)
            order.append(i)

        visit(cid)
        lines = []
        for i in order:
            origin = parents[i]
            how = origin[0]
            if len(origin) > 1:
                how += f"({', '.join(map(str, origin[1:]))})"
            lines.append(f"[{i}] {clause_str(store[i])}    <- {how}")
        return lines


def prove_formula(premises, goal, prover: Prover | None = None) -> ProofResult:
    """Does the conjunction of FOL ``premises`` entail ``goal``?"""
    from reasoning_agent.logic import _Skolemizer

    prover = prover or Prover()
    sk = _Skolemizer()
    axioms = []
    for f in premises:
        axioms.extend(to_clauses(f, sk))
    negated = to_clauses(("not", goal), _Skolemizer("skg"))
    return prover.refute(axioms, negated)
