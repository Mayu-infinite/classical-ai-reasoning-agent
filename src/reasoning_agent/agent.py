"""End-to-end reasoning agent: (context, query) -> True / False.

Pipeline
    natural language / FOL text
        -> representation (Horn rules, FOL clauses, or a causal model)
        -> classical inference (forward/backward chaining, resolution,
           counterfactual model checking)
        -> label

Routing
    * FOL annotations (or FOL symbols in the context) -> resolution prover.
    * Causal language + counterfactual question     -> structural causal model.
    * English that compiles to Horn rules            -> forward chaining with
      negation as failure (closed-world, well-founded semantics).
    * English with disjunctive / existential content -> resolution prover
      (open-world).
"""

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from reasoning_agent.causal import answer_causal, looks_causal
from reasoning_agent.datalog import BackwardChainer, ForwardChainer, Rule, explain
from reasoning_agent.english import EnglishParser, ParseFailure, split_sentences
from reasoning_agent.logic import (
    Var,
    close_universally,
    literal_str,
    parse_fol,
    repair_symbols,
)
from reasoning_agent.prover import Prover, prove_formula

FOL_SYMBOLS = re.compile(r"[∀∃→∧∨¬⊕↔]")


class _CompiledKB:
    """A parsed knowledge base plus lazily computed inference structures."""

    def __init__(self, parser: EnglishParser, items, failures):
        self.parser = parser
        self.failures = failures
        self.rules = [i for i in items if isinstance(i, Rule)]
        self.formulas = [i[1] for i in items if not isinstance(i, Rule)]
        self.controlled = not failures and EnglishParser.is_controlled(items)
        self.constants = {a for r in self.rules for _, _, args in (*r.body, r.head)
                          for a in args if type(a) is not Var}
        self.model = None
        self.chainer = None


@dataclass
class Prediction:
    label: bool
    engine: str
    trace: list = field(default_factory=list)
    seconds: float = 0.0
    detail: str = ""


class ReasoningAgent:
    def __init__(self, strategy: str = "forward", prover_time_limit: float = 0.3,
                 closed_world: bool = True, use_fol_annotations: bool = True,
                 cache_size: int = 4096):
        if strategy not in ("forward", "backward"):
            raise ValueError("strategy must be 'forward' or 'backward'")
        self.strategy = strategy
        self.closed_world = closed_world
        self.use_fol = use_fol_annotations
        self.prover = Prover(time_limit=prover_time_limit)
        self.cache_size = cache_size
        self._kb_cache: OrderedDict[str, _CompiledKB] = OrderedDict()

    # -- entry point --------------------------------------------------------------

    def predict(self, context: str, query: str, premises_fol: str | None = None,
                conclusion_fol: str | None = None, explain_proof: bool = False) -> Prediction:
        start = time.perf_counter()
        try:
            if self.use_fol and premises_fol and conclusion_fol:
                pred = self._fol(premises_fol, conclusion_fol)
            elif FOL_SYMBOLS.search(context or "") and FOL_SYMBOLS.search(query or "") or (
                FOL_SYMBOLS.search(context or "") and _looks_like_fol(query)
            ):
                pred = self._fol(context, query)
            elif looks_causal(context, query):
                pred = self._causal(context, query)
            else:
                pred = self._english(context, query, explain_proof)
        except Exception as exc:  # never crash on a single malformed example
            pred = Prediction(False, "fallback", [f"error: {type(exc).__name__}: {exc}"])
        pred.seconds = time.perf_counter() - start
        return pred

    # -- engines ------------------------------------------------------------------

    def _fol(self, premises_text: str, conclusion_text: str) -> Prediction:
        if not isinstance(premises_text, str):
            premises_text = "\n".join(premises_text)
        premises = []
        skipped = 0
        for line in re.split(r"\n|(?<=\))\s*;\s*", premises_text):
            if line.strip():
                try:
                    premises.append(parse_fol(line))
                except ValueError:
                    skipped += 1
        goal = parse_fol(conclusion_text, free_quantifier="ex")
        premises, goal, repairs = repair_symbols(premises, goal)
        result = prove_formula(premises, goal, self.prover)
        trace = [f"premises: {len(premises)} formulas ({skipped} unparsable)"]
        if repairs:
            trace.append("symbol repairs: " + ", ".join(f"{k}->{v}" for k, v in repairs.items()))
        trace.append(f"prover: {result.reason}, {result.steps} given clauses, "
                     f"{result.clauses} kept")
        trace += result.proof[-25:]
        return Prediction(result.proved, "resolution", trace, detail=result.reason)

    def _causal(self, context: str, query: str) -> Prediction:
        ans = answer_causal(context, query)
        trace = ans.trace + [f"P(outcome) = {ans.probability:.3f} over {ans.worlds} world(s)"]
        return Prediction(ans.answer, "causal", trace)

    def _compile(self, context: str) -> "_CompiledKB":
        """Parse a context once; repeated queries on the same KB reuse it."""
        kb = self._kb_cache.get(context)
        if kb is not None:
            self._kb_cache.move_to_end(context)
            return kb
        parser = EnglishParser()
        items, failures = parser.parse_text(context)
        parser.last_entity = None
        kb = _CompiledKB(parser, items, failures)
        self._kb_cache[context] = kb
        if len(self._kb_cache) > self.cache_size:
            self._kb_cache.popitem(last=False)
        return kb

    def _english(self, context: str, query: str, explain_proof: bool) -> Prediction:
        kb = self._compile(context)
        try:
            q_items = kb.parser.fork().parse_sentence(query)
        except ParseFailure as exc:
            return Prediction(False, "fallback", [f"query not understood: {exc}"])
        q_rules = [i for i in q_items if isinstance(i, Rule)]
        q_formulas = [i[1] for i in q_items if not isinstance(i, Rule)]
        notes = [f"parsed {len(kb.rules)} rules/facts, {len(kb.formulas)} formulas, "
                 f"{len(kb.failures)} unparsed sentences"]
        notes += [f"unparsed: {s!r} ({e})" for s, e in kb.failures[:5]]
        if not q_rules and not q_formulas:
            return Prediction(False, "fallback", notes + ["empty query"])

        if kb.controlled and EnglishParser.is_controlled(q_items) and self.closed_world:
            return self._horn(kb, q_rules, notes, explain_proof)
        notes.append("free-form text: open-world semantics")
        return self._open_world(kb.rules, kb.formulas, q_rules, q_formulas, notes)

    def _horn(self, kb, q_rules, notes, explain_proof) -> Prediction:
        rules = list(kb.rules)
        goals = []

        def sub(args):
            return tuple("hypothetical_individual" if type(a) is Var else a for a in args)

        hypothetical = False
        for q in q_rules:
            if q.body:
                # A rule-shaped query: hypothetically assert the body for a
                # fresh individual and test whether the head follows.
                rules += [Rule((), (p, n, sub(a))) for p, n, a in q.body]
                goals.append((q.head[0], q.head[1], sub(q.head[2])))
                hypothetical = True
            else:
                goals.append(q.head)
        if any(type(a) is Var for g in goals for a in g[2]):
            return Prediction(False, "fallback", notes + ["non-ground query"])

        domain = {a for g in goals for a in g[2]}
        # The cached model is valid when the query adds no rules and no new
        # individuals (negation as failure quantifies over the domain).
        reusable = not hypothetical and domain <= kb.constants
        if self.strategy == "backward":
            if reusable:
                kb.chainer = kb.chainer or BackwardChainer(rules, (), cwa=True)
                bc = kb.chainer
            else:
                bc = BackwardChainer(rules, domain, cwa=True)
            label = all(bc.prove(g) for g in goals)
            trace = notes + [f"backward chaining: {bc.calls} goal calls"]
            return Prediction(label, "backward-chaining", trace)

        if reusable:
            kb.model = kb.model or ForwardChainer(rules, (), cwa=True).model()
            model = kb.model
        else:
            model = ForwardChainer(rules, domain, cwa=True).model()
        label = all(model.holds(*g) for g in goals)
        trace = notes + [f"forward chaining: {len(model.true)} true atoms, "
                         f"{model.iterations} alternating-fixpoint round(s), "
                         f"{model.rule_firings} rule firings"]
        if explain_proof:
            for g in goals:
                trace += explain(model, g)
        else:
            trace += [f"query {literal_str(g)}: {'holds' if model.holds(*g) else 'fails'}"
                      for g in goals]
        return Prediction(label, "forward-chaining", trace)

    def _open_world(self, rules, formulas, q_rules, q_formulas, notes) -> Prediction:
        premises = [_rule_formula(r) for r in rules] + [close_universally(f) for f in formulas]
        parts = [_rule_formula(r) for r in q_rules] + [close_universally(f) for f in q_formulas]
        goal = parts[0] if len(parts) == 1 else ("and", *parts)
        result = prove_formula(premises, goal, self.prover)
        trace = notes + [f"resolution: {result.reason}"] + result.proof[-15:]
        return Prediction(result.proved, "resolution", trace, detail=result.reason)


def _looks_like_fol(text: str) -> bool:
    return bool(re.match(r"^\s*¬?\s*[A-Za-z]\w*\([\w\s,]+\)\s*$", text or ""))


def _rule_formula(rule: Rule):
    def lit(literal):
        positive, pred, args = literal
        atom = ("atom", pred, args)
        return atom if positive else ("not", atom)

    head = lit(rule.head)
    f = head if not rule.body else ("imp", ("and", *(lit(b) for b in rule.body)), head)
    return close_universally(f)


def explain_text(agent: ReasoningAgent, context: str, query: str, max_context=300) -> str:
    pred = agent.predict(context, query, explain_proof=True)
    shown = " ".join(split_sentences(context))
    if max_context and len(shown) > max_context:
        shown = shown[:max_context] + " ..."
    lines = [f"Context: {shown}", f"Query: {query}",
             f"Engine: {pred.engine}", f"Prediction: {pred.label}"]
    lines += ["  " + t for t in pred.trace]
    return "\n".join(lines)
