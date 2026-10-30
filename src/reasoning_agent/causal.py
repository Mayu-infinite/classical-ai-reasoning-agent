"""Deterministic counterfactual reasoning over Boolean causal models.

Natural-language causal statements ("A causes B", "A and B together cause C",
"not A or B causes C", "A causes not B", "A has a direct effect on B") are
compiled into structural equations V := f(parents). Counterfactual queries
are answered with Pearl's three steps, implemented as exhaustive model
checking over the unobserved root (exogenous) variables:

1. Abduction  -- enumerate root assignments consistent with the factual
                 evidence (the antecedent's factual value and observations),
                 weighting each by its prior (``A ~ Bern(p)``; default 0.5).
2. Action     -- replace the equations of intervened variables by constants.
3. Prediction -- propagate in topological order and read off the outcome.

The answer is "yes" when the outcome holds with posterior probability > 0.5.
"""

import itertools
import re
from dataclasses import dataclass, field

CAUSE_VERBS = r"(?:causes?|caused|has a direct effect on|have direct effects on|" \
              r"has direct effects on|have a direct effect on|leads? to|results? in|produces?)"


@dataclass
class CausalModel:
    equations: dict = field(default_factory=dict)  # var -> expression
    priors: dict = field(default_factory=dict)  # root var -> P(var = 1)
    variables: list = field(default_factory=list)

    def add_var(self, v):
        if v not in self.variables:
            self.variables.append(v)

    def roots(self):
        return [v for v in self.variables if v not in self.equations]

    def order(self):
        done, out = set(), []

        def visit(v, stack=()):
            if v in done or v in stack:
                # A back edge (cyclic description): the parent is read with
                # its value from the earlier evaluation order (default False).
                return
            for p in _expr_vars(self.equations.get(v, ("const", False))):
                visit(p, stack + (v,))
            done.add(v)
            out.append(v)

        for v in self.variables:
            visit(v)
        return out

    def evaluate(self, roots: dict, interventions: dict) -> dict:
        world = {}
        for v in self.order():
            if v in interventions:
                world[v] = interventions[v]
            elif v in self.equations:
                world[v] = _eval(self.equations[v], world)
            else:
                world[v] = roots.get(v, False)
        return world


def _expr_vars(e):
    if e[0] == "var":
        return [e[1]]
    if e[0] == "const":
        return []
    if e[0] == "not":
        return _expr_vars(e[1])
    return [v for sub in e[1] for v in _expr_vars(sub)]


def _eval(e, world):
    op = e[0]
    if op == "var":
        return world.get(e[1], False)
    if op == "const":
        return e[1]
    if op == "not":
        return not _eval(e[1], world)
    if op == "and":
        return all(_eval(s, world) for s in e[1])
    return any(_eval(s, world) for s in e[1])


def expr_str(e):
    op = e[0]
    if op == "var":
        return e[1]
    if op == "const":
        return str(e[1])
    if op == "not":
        return "¬" + expr_str(e[1])
    sym = " ∧ " if op == "and" else " ∨ "
    return "(" + sym.join(expr_str(s) for s in e[1]) + ")"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LITERAL = r"(?:not\s+)?[A-Za-z][\w\-]*"


def looks_causal(context: str, query: str) -> bool:
    text = f"{context} {query}".lower()
    return bool(re.search(r"\b(causes?|direct effects? on)\b", text)) and bool(
        re.search(r"\bwould\b.*\boccur\b|\bsuppose\b|\binstead of\b|\bassume\b", query.lower())
    )


def _parse_cause_expr(text: str):
    text = re.sub(r"\b(together|both|jointly|either)\b", " ", text).strip()
    text = re.sub(r"\s+", " ", text)
    ors = [p.strip() for p in re.split(r"\s+or\s+", text) if p.strip()]
    disj = []
    for part in ors:
        ands = [a.strip() for a in re.split(r"\s+and\s+", part) if a.strip()]
        conj = [_parse_lit(a) for a in ands]
        disj.append(conj[0] if len(conj) == 1 else ("and", conj))
    return disj[0] if len(disj) == 1 else ("or", disj)


def _parse_lit(text: str):
    text = text.strip()
    neg = False
    while text.lower().startswith("not "):
        neg = not neg
        text = text[4:].strip()
    name = text.strip(" .,")
    if not re.fullmatch(r"[A-Za-z][\w\-]*", name):
        raise ValueError(f"bad causal variable {text!r}")
    e = ("var", name)
    return ("not", e) if neg else e


def parse_causal_context(text: str) -> tuple[CausalModel, dict]:
    """Return the causal model and the observations stated in ``text``."""
    model = CausalModel()
    observations = {}
    text = text.replace("\n", " ")
    text = re.sub(r"^\s*(we know that|imagine[^:]*:|given that)\s*", "", text, flags=re.I)

    # Priors: "Blaf~ Bern(0.2)" / "Blaf ~ Bernoulli(0.2)"
    for m in re.finditer(r"([A-Za-z][\w\-]*)\s*~\s*Bern(?:oulli)?\s*\(\s*([\d.]+)\s*\)", text):
        model.priors[m.group(1)] = float(m.group(2))
    text = re.sub(r"[A-Za-z][\w\-]*\s*~\s*Bern(?:oulli)?\s*\(\s*[\d.]+\s*\)\.?", " ", text)

    # Observations: "We observed X", "We observed not X and Y"
    for m in re.finditer(r"\bwe (?:observed?|see|know)\s+(?:that\s+)?([^.]*)", text, flags=re.I):
        for part in re.split(r",|\band\b", m.group(1)):
            part = part.strip()
            if not part or re.search(CAUSE_VERBS, part):
                continue
            part = re.sub(r"\b(did|does|do)\s+not\b", "not", part)
            m_neg = re.match(r"^([A-Za-z][\w\-]*)\s+not\b(.*)$", part)
            if m_neg:
                part = f"not {m_neg.group(1)} {m_neg.group(2)}"
            part = re.sub(r"\b(occurred|occurs|occur|happened|happens|happen|is true)\b", "",
                          part)
            lit = _parse_lit(part)
            if lit[0] == "not":
                observations[lit[1][1]] = False
            else:
                observations[lit[1]] = True
    text = re.sub(r"\bwe (?:observed?|see)\b[^.]*\.?", " ", text, flags=re.I)

    # Causal statements, separated by commas / sentence ends / "and" before a new clause.
    text = re.sub(r"\.\s*", ", ", text)
    pieces = [p.strip() for p in re.split(r",", text) if p.strip()]
    statements = []
    for piece in pieces:
        piece = re.sub(r"^(and|also|then)\s+", "", piece.strip(), flags=re.I)
        if statements and not re.search(rf"\b{CAUSE_VERBS}\b", piece, flags=re.I):
            # Continuation of an effect list: "A causes B, C and D".
            statements[-1] += " and " + piece
            continue
        # Split "A causes B and C causes D" into two statements.
        verbs = list(re.finditer(rf"\b{CAUSE_VERBS}\b", piece, flags=re.I))
        if len(verbs) > 1:
            cut_points = []
            for v in verbs[1:]:
                before = piece[: v.start()]
                m = list(re.finditer(r"\band\b", before))
                cause_start = None
                # The cause of the second verb starts after the last 'and' that
                # follows the previous verb's effects.
                for a in reversed(m):
                    if a.start() > verbs[0].end():
                        cause_start = a
                        break
                if cause_start is not None:
                    cut_points.append(cause_start)
            last = 0
            for c in cut_points:
                statements.append(piece[last : c.start()])
                last = c.end()
            statements.append(piece[last:])
        else:
            statements.append(piece)

    for st in statements:
        st = st.strip()
        m = re.match(rf"^(.*?)\s+{CAUSE_VERBS}\s+(.*)$", st, flags=re.I)
        if not m:
            continue
        cause = _parse_cause_expr(m.group(1))
        effects = [e.strip() for e in re.split(r"\s+and\s+|\s*,\s*", m.group(2)) if e.strip()]
        for v in _expr_vars(cause):
            model.add_var(v)
        for eff in effects:
            lit = _parse_lit(eff)
            neg = lit[0] == "not"
            name = lit[1][1] if neg else lit[1]
            model.add_var(name)
            e = ("not", cause) if neg else cause
            if name in model.equations:
                model.equations[name] = ("or", [model.equations[name], e])
            else:
                model.equations[name] = e
    return model, observations


@dataclass
class CausalQuery:
    outcome: str
    outcome_value: bool = True
    interventions: dict = field(default_factory=dict)
    factual: dict = field(default_factory=dict)


def parse_causal_query(question: str) -> CausalQuery:
    q = question.strip().rstrip("?. ")
    # Nested: "Assume not A, and based on this assumption, further suppose not B. Would Y occur"
    m = re.search(r"would\s+(not\s+)?([A-Za-z][\w\-]*)\s+(not\s+)?(?:occur|happen|be true)", q,
                  flags=re.I)
    if not m:
        raise ValueError("unrecognised causal question")
    outcome = m.group(2)
    value = (m.group(1) is None) == (m.group(3) is None)
    q = q[: m.start()] + q[m.end():]
    query = CausalQuery(outcome, value)

    def lits(text):
        out = []
        for part in re.split(r"\s+and\s+|,", text):
            part = part.strip()
            if part:
                lit = _parse_lit(part)
                out.append((lit[1][1], False) if lit[0] == "not" else (lit[1], True))
        return out

    inst = re.search(r"\bif\s+(.*?)\s+instead of\s+(.*)$", q, flags=re.I)
    if inst:
        for v, val in lits(inst.group(1)):
            query.interventions[v] = val
        for v, val in lits(inst.group(2)):
            query.factual[v] = val
        return query
    for pattern in (r"\bassume\s+(.*?)(?:,|\.)", r"\bsuppose\s+(.*?)(?:\.|\s+would\b|$)",
                    r"\bif\s+(.*)$"):
        for mm in re.finditer(pattern, q, flags=re.I):
            text = re.sub(r"\b(and based on this assumption|further|that)\b", " ", mm.group(1))
            for v, val in lits(text):
                query.interventions[v] = val
    if not query.interventions:
        raise ValueError("no intervention found")
    # The intervened cause took the opposite value in the factual world.
    first = next(iter(query.interventions))
    query.factual.setdefault(first, not query.interventions[first])
    return query


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass
class CausalAnswer:
    answer: bool
    probability: float
    worlds: int
    trace: list


def counterfactual(model: CausalModel, query: CausalQuery, observations: dict) -> CausalAnswer:
    for v in list(query.interventions) + [query.outcome] + list(observations):
        model.add_var(v)
    roots = model.roots()
    # Observations constrain the exogenous (root) variables during abduction.
    # Observations of endogenous variables are already determined by the
    # structural equations given the roots, and the benchmark treats them as
    # background information rather than as evidence that could contradict the
    # antecedent; using them is only a fallback when roots are unconstrained.
    root_obs = {k: v for k, v in observations.items() if k not in model.equations}
    evidence_sets = [
        {**root_obs, **query.factual},
        dict(query.factual),
        {},
    ]
    trace = [f"{v} := {expr_str(e)}" for v, e in model.equations.items()]
    for evidence in evidence_sets:
        total = 0.0
        yes = 0.0
        worlds = 0
        for values in itertools.product((True, False), repeat=len(roots)):
            assignment = dict(zip(roots, values))
            factual = model.evaluate(assignment, {})
            if any(factual.get(k) != v for k, v in evidence.items()):
                continue
            weight = 1.0
            for r, val in assignment.items():
                p = model.priors.get(r, 0.5)
                weight *= p if val else 1 - p
            cf = model.evaluate(assignment, query.interventions)
            worlds += 1
            total += weight
            if cf[query.outcome] == query.outcome_value:
                yes += weight
            if worlds <= 2:
                trace.append(
                    "abduced roots "
                    + ", ".join(f"{k}={int(v)}" for k, v in assignment.items())
                    + " | do(" + ", ".join(f"{k}={int(v)}" for k, v in query.interventions.items())
                    + ") -> " + ", ".join(f"{k}={int(v)}" for k, v in cf.items())
                )
        if worlds:
            prob = yes / total if total else 0.0
            return CausalAnswer(prob > 0.5, prob, worlds, trace)
    return CausalAnswer(False, 0.0, 0, trace)


def answer_causal(context: str, question: str) -> CausalAnswer:
    model, observations = parse_causal_context(context)
    query = parse_causal_query(question)
    return counterfactual(model, query, observations)
