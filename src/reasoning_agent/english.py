"""Rule-based semantic parser: restricted English -> logic.

The parser is compositional rather than template based. A sentence is
segmented into a *condition* and a *conclusion* (``if/when/whenever ...``,
``... if ...``, ``all/every/no/some N ...``, generic plurals), each side is
split into coordinated clauses, and every clause is analysed as

    SUBJECT  VERB-PHRASE

where the subject is an entity (``the bald eagle``, ``Anne``), an indefinite
or pronoun (``someone``, ``they``, ``it``) bound to a rule variable, and the
verb phrase is a copula + adjective/noun complement, a (negated) transitive or
intransitive verb, or a ``has``/``can`` construction.

Output items are ``Rule`` objects (Horn rules / facts, possibly with negated
literals) or general FOL formulas for disjunctive/existential content.
"""

import re
from dataclasses import dataclass, field

from reasoning_agent.datalog import Rule
from reasoning_agent.logic import Var

COPULA = {"is", "are", "am", "was", "were", "be", "being", "been"}
AUX = {"does", "do", "did"}
MODALS = {"can", "could", "will", "would", "shall", "should", "may", "might", "must"}
NEGATORS = {"not", "never", "no"}
DETERMINERS = {"a", "an", "the"}
QUANTIFIERS = {"all", "every", "each", "any", "no", "some"}
RELATIVE = {"who", "that", "which", "whom"}
PRONOUNS = {"they", "it", "he", "she", "them", "him", "her", "itself", "themselves",
            "himself", "herself"}
INDEFINITES = {"someone", "something", "somebody", "anyone", "anything", "anybody",
               "everyone", "everything", "everybody", "one"}
GENERIC_NOUNS = {"thing", "things", "people", "person", "persons", "someone", "something",
                 "somebody", "one", "ones", "individual", "individuals", "everyone",
                 "everything", "everybody", "anyone", "anything", "anybody", "guy",
                 "guys", "entity", "entities", "creature", "creatures", "being",
                 "beings", "object", "objects", "animal", "animals", "man", "men",
                 "woman", "women"}
FILLERS = {"always", "usually", "normally", "generally", "typically", "often", "also",
           "really", "very", "quite", "rather", "pretty", "definitely", "certainly",
           "surely", "indeed", "just", "too", "actually", "truly", "sometimes",
           "absolutely", "extremely", "somewhat", "fairly", "likewise",
           "still", "necessarily", "therefore", "thus", "hence", "so"}
COPULA_PHRASES = [
    ("tends to be", "is"), ("tend to be", "are"), ("seems to be", "is"),
    ("seem to be", "are"), ("is going to be", "is"), ("are going to be", "are"),
    ("will be", "is"), ("must be", "is"), ("should be", "is"), ("would be", "is"),
    ("is sure to be", "is"), ("are sure to be", "are"), ("happens to be", "is"),
    ("happen to be", "are"), ("is considered", "is"), ("are considered", "are"),
    ("is likely to be", "is"), ("are likely to be", "are"), ("can be", "is"),
    ("turns out to be", "is"), ("turn out to be", "are"), ("ends up being", "is"),
    ("is also", "is"), ("are also", "are"), ("is definitely", "is"),
]
_COPULA_PHRASE_WORDS: dict[str, list] = {}
for _phrase, _repl in sorted(COPULA_PHRASES, key=lambda x: -len(x[0].split())):
    _COPULA_PHRASE_WORDS.setdefault(_phrase.split()[0], []).append((_phrase.split(), _repl))
CONTRACTIONS = {
    "doesn't": "does not", "don't": "do not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "can't": "can not", "cannot": "can not", "won't": "will not",
    "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
    "hasn't": "has not", "haven't": "have not", "it's": "it is", "they're": "they are",
    "he's": "he is", "she's": "she is", "someone's": "someone is",
}
PREPOSITIONS = {"of", "to", "with", "at", "in", "on", "for", "from", "by", "about",
                "into", "than", "as", "like", "near", "under", "over"}


_FUNCTION_WORDS = (COPULA | AUX | MODALS | DETERMINERS | QUANTIFIERS | PRONOUNS | INDEFINITES
                   | {"if", "when", "whenever", "there", "everyone", "nobody", "nothing",
                      "either", "neither", "only"})


class ParseFailure(ValueError):
    pass


# ---------------------------------------------------------------------------
# Lexical helpers
# ---------------------------------------------------------------------------


def lemma_verb(word: str) -> str:
    """Third-person singular / plural verb forms -> one canonical form."""
    w = word.lower()
    if w in ("is", "are", "am", "was", "were", "be"):
        return "be"
    if w in ("has", "have", "had"):
        return "have"
    if w in ("does", "do", "did"):
        return "do"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zzes", "oes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 2:
        return w[:-1]
    return w


def singular(noun: str) -> str:
    n = noun.lower()
    irregular = {"people": "person", "men": "man", "women": "woman", "mice": "mouse",
                 "geese": "goose", "children": "child", "teeth": "tooth", "feet": "foot",
                 "oxen": "ox", "cacti": "cactus", "wolves": "wolf", "knives": "knife",
                 "lives": "life", "leaves": "leaf", "halves": "half", "calves": "calf"}
    if n in irregular:
        return irregular[n]
    if n.endswith("ies") and len(n) > 4:
        return n[:-3] + "y"
    if n.endswith(("ches", "shes", "sses", "xes", "zes")):
        return n[:-2]
    if n.endswith("s") and not n.endswith(("ss", "us", "is", "ous")) and len(n) > 3:
        return n[:-1]
    return n


def looks_plural(word: str) -> bool:
    w = word.lower()
    return w in ("people", "men", "women", "mice", "children", "geese") or (
        w.endswith("s") and not w.endswith(("ss", "us", "ous", "is")) and len(w) > 3
    )


def entity_name(words: list[str]) -> str:
    return "_".join(w.lower() for w in words)


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


@dataclass
class Tok:
    text: str  # lower-case
    cap: bool  # capitalised in the source


def normalize(sentence: str) -> list[Tok]:
    s = sentence.strip()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[.!?;:]+\s*$", "", s)
    words = re.findall(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*|,", s)
    expanded: list[Tok] = []
    for w in words:
        low = w.lower()
        if low in CONTRACTIONS:
            for part in CONTRACTIONS[low].split():
                expanded.append(Tok(part, False))
        elif low.endswith("n't"):
            expanded.append(Tok(low[:-3], w[0].isupper()))
            expanded.append(Tok("not", False))
        else:
            expanded.append(Tok(low, w[0].isupper()))
    if expanded:
        expanded[0].cap = expanded[0].cap and expanded[0].text not in _FUNCTION_WORDS
    # Multi-word copula paraphrases ("tends to be" -> "is").
    out: list[Tok] = []
    i = 0
    while i < len(expanded):
        for phrase, repl in _COPULA_PHRASE_WORDS.get(expanded[i].text, ()):
            if [t.text for t in expanded[i : i + len(phrase)]] == phrase:
                out.append(Tok(repl, False))
                i += len(phrase)
                break
        else:
            if expanded[i].text not in FILLERS:
                out.append(expanded[i])
            i += 1
    return out


# ---------------------------------------------------------------------------
# Parsing context
# ---------------------------------------------------------------------------


@dataclass
class Ctx:
    """Variable bindings for one sentence."""

    main: object = None  # the rule's primary variable (or entity)
    others: list = field(default_factory=list)
    last_entity: str | None = None
    vars_: dict = field(default_factory=dict)

    def variable(self, key="x"):
        if key not in self.vars_:
            self.vars_[key] = Var(key)
        return self.vars_[key]


@dataclass
class Clause:
    subject: object
    lits: list  # list of literals (positive, pred, args)
    kind: str = "conj"  # "conj", "or" (disjunction of lits), "nor"
    copular: bool = False
    negated: bool = False
    verb: str | None = None
    indefinite: bool = False


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------


class EnglishParser:
    def __init__(self):
        self.known_verbs: set[str] = set()
        self.known_entities: set[str] = set()
        self.last_entity: str | None = None
        self._pending_verbs: set[str] = set()
        self._lemma_cache: tuple = (set(), set())
        self._witnesses = 0

    # -- public API -------------------------------------------------------------

    def fork(self) -> "EnglishParser":
        """A copy sharing the learnt lexicon but not mutating this parser."""
        other = EnglishParser()
        other.known_verbs = set(self.known_verbs)
        other.known_entities = set(self.known_entities)
        other.last_entity = self.last_entity
        other._witnesses = self._witnesses
        return other

    @staticmethod
    def is_controlled(items) -> bool:
        """True if every item uses only the core controlled-language fragment.

        Fallback constructions (prepositional / multi-word predicates,
        equality, disjunctions) signal free-form text, for which the closed
        world assumption is not appropriate.
        """
        for it in items:
            if not isinstance(it, Rule):
                return False
            for _, pred, _ in (*it.body, it.head):
                if "_" in pred or pred == "=":
                    return False
        return True

    def parse_text(self, text: str):
        """Parse a multi-sentence context. Returns (items, failures)."""
        items, failures = [], []
        for sentence in split_sentences(text):
            try:
                items.extend(self.parse_sentence(sentence))
            except ParseFailure as exc:
                failures.append((sentence, str(exc)))
        return items, failures

    def parse_sentence(self, sentence: str):
        toks = normalize(sentence)
        if not toks:
            return []
        self._pending_verbs = set()
        items = self._sentence(toks)
        # Verbs are only learnt from clauses of successfully parsed sentences
        # that produced binary relations.
        binary = {q for it in items if isinstance(it, Rule)
                  for _, q, a in (*it.body, it.head) if len(a) == 2}
        self.known_verbs |= {v for v in self._pending_verbs if lemma_verb(v) in binary}
        for item in items:
            item_text = sentence.strip()
            if isinstance(item, Rule):
                object.__setattr__(item, "text", item_text)
        return items

    # -- sentence level ---------------------------------------------------------

    def _sentence(self, toks):
        words = [t.text for t in toks]
        ctx = Ctx(last_entity=self.last_entity)
        if words[0] in ("if", "when", "whenever", "provided") or words[:2] == ["as", "long"]:
            start = 1 if words[0] != "as" else 3
            body_toks, head_toks = self._split_condition(toks[start:])
            return self._conditional(body_toks, head_toks, ctx)
        if "if" in words[1:]:
            k = words.index("if", 1)
            if words[k - 1] not in ("even", "as", "only"):
                head = toks[:k]
                if head and head[-1].text == ",":
                    head = head[:-1]
                body = toks[k + 1 :]
                if self._starts_with_entity(head) or head[0].text in PRONOUNS | INDEFINITES:
                    return self._conditional(body, head, ctx)
                # Generic head: "Big people are red if they are young."
                rules = self._quantified(head, ctx, implicit=head[0].text not in QUANTIFIERS)
                extra = self._condition(body, ctx)
                return [Rule(r.body + tuple(alt), r.head) for r in rules
                        if isinstance(r, Rule) for alt in extra]
        if len(toks) > 3 and toks[0].cap and words[1] == "," and words[2] in RELATIVE:
            # Appositive: "Alan, who is round and red, is blue" -> facts.
            toks = [toks[0]] + toks[3:]
            words = [t.text for t in toks]
        if words[0] == "there" and len(words) > 2 and words[1] in ("is", "are", "exists", "exist"):
            return self._existential(toks[2:], ctx)
        if words[0] in QUANTIFIERS or words[0] in ("everyone", "everything", "everybody",
                                                   "nobody", "nothing", "anyone", "anything"):
            return self._quantified(toks, ctx)
        if words[0] in ("a", "an"):
            return self._quantified([Tok("every", False)] + toks[1:], ctx)
        if words[0] in INDEFINITES:
            return self._quantified(toks, ctx)
        if not self._starts_with_entity(toks):
            return self._quantified(toks, ctx, implicit=True)
        clauses = self._conjunction(toks, ctx, None)
        return self._facts(clauses, ctx)

    def _split_condition(self, toks):
        words = [t.text for t in toks]
        if "then" in words:
            k = words.index("then")
            body = toks[:k]
            if body and body[-1].text == ",":
                body = body[:-1]
            return body, toks[k + 1 :]
        # "If A, B": split at the last comma whose right side starts a clause.
        commas = [i for i, w in enumerate(words) if w == ","]
        for k in reversed(commas):
            right = toks[k + 1 :]
            if right and (right[0].text in PRONOUNS | INDEFINITES | {"the"} or right[0].cap):
                return toks[:k], right
        if commas:
            k = commas[-1]
            return toks[:k], toks[k + 1 :]
        # "If someone is red they are big": split before a pronoun subject.
        for k in range(1, len(words)):
            if words[k] in ("they", "it", "he", "she") and k + 1 < len(words) and (
                words[k + 1] in COPULA | AUX | MODALS or self._is_verb_word(words[k + 1])
            ):
                return toks[:k], toks[k:]
        raise ParseFailure("cannot split condition / conclusion")

    def _conditional(self, body_toks, head_toks, ctx):
        alternatives = self._condition(body_toks, ctx)
        heads = self._conjunction(head_toks, ctx, None, head=True)
        rules = []
        for alt in alternatives:
            rules.extend(self._make_rules(alt, heads, ctx))
        return rules

    def _condition(self, toks, ctx):
        """Condition -> list of alternatives, each a list of literals (DNF)."""
        words = [t.text for t in toks]
        groups = _split_top(toks, "or") if "or" in words and "either" not in words else [toks]
        alternatives = []
        prev = None
        for group in groups:
            clauses = self._conjunction(group, ctx, prev)
            prev = clauses[-1] if clauses else prev
            lits = []
            for c in clauses:
                if c.kind == "or":
                    raise ParseFailure("disjunction inside condition clause")
                lits.extend(c.lits)
            alternatives.append(lits)
        return alternatives

    def _make_rules(self, body_lits, head_clauses, ctx):
        rules = []
        for c in head_clauses:
            if c.kind == "or":
                return [("formula", _implication(body_lits, c))]
            for lit in c.lits:
                rules.append(Rule(tuple(body_lits), lit))
        return rules

    def _facts(self, clauses, ctx):
        items = []
        for c in clauses:
            if isinstance(c.subject, str):
                self.last_entity = c.subject
            if c.kind == "or":
                items.append(("formula", ("or", *(_lit_formula(lit) for lit in c.lits))))
                continue
            for lit in c.lits:
                items.append(Rule((), lit))
        return items

    def _existential(self, toks, ctx):
        """'There is a big dog that is red' -> skolem constant facts."""
        restrictor, vp = self._split_np_vp(toks, ctx, allow_no_vp=True)
        lits = restrictor
        if vp:
            for c in vp:
                lits = lits + c.lits
        self._witnesses += 1
        name = f"witness_{self._witnesses}"
        x = ctx.main
        return [Rule((), (p, q, tuple(name if a is x else a for a in args))) for p, q, args in lits]

    def _quantified(self, toks, ctx, implicit=False):
        words = [t.text for t in toks]
        quant = None
        if not implicit:
            first = words[0]
            if first in ("everyone", "everything", "everybody", "anyone", "anything"):
                quant = "all"
                toks = [Tok("all", False), Tok("things", False)] + toks[1:]
            elif first in ("nobody", "nothing"):
                quant = "no"
                toks = [Tok("no", False), Tok("things", False)] + toks[1:]
            elif first in ("someone", "something", "somebody"):
                quant = "all"
                toks = [Tok("all", False), Tok("things", False)] + toks[1:]
            else:
                quant = {"every": "all", "each": "all", "any": "all"}.get(first, first)
            toks = toks[1:]
        else:
            quant = "all"
        if quant == "some":
            return self._existential(toks, ctx)
        restrictor, heads = self._split_np_vp(toks, ctx)
        if quant == "no":
            negated = []
            for c in heads:
                if c.kind != "conj" or len(c.lits) != 1:
                    return [("formula", _negated_universal(restrictor, c))]
                p, q, a = c.lits[0]
                negated.append(Clause(c.subject, [(not p, q, a)]))
            heads = negated
        return self._make_rules(restrictor, heads, ctx)

    def _split_np_vp(self, toks, ctx, allow_no_vp=False):
        """Split 'ADJ* NOUN [who VP] VP' into restrictor literals + head clauses."""
        x = ctx.variable("x")
        ctx.main = x
        # Prefer splitting at a copula/auxiliary, then at a lexical verb.
        candidates = [i for i, t in enumerate(toks) if i > 0 and t.text in COPULA | AUX | MODALS]
        candidates += [
            i for i, t in enumerate(toks)
            if i > 0 and i not in candidates
            and (self._is_verb_word(t.text) or t.text in ("has", "have"))
        ]
        errors = []
        for k in candidates:
            try:
                restrictor = self._restrictor(toks[:k], x, ctx)
                heads = self._conjunction(toks[k:], ctx, Clause(x, [], copular=False),
                                          head=True, subject=x)
                return restrictor, heads
            except ParseFailure as exc:
                errors.append(str(exc))
        if allow_no_vp:
            return self._restrictor(toks, x, ctx), []
        raise ParseFailure("cannot split noun phrase and verb phrase: " + "; ".join(errors[:2]))

    def _restrictor(self, toks, x, ctx):
        """'kind , smart things who are big' -> literals over x."""
        words = [t.text for t in toks]
        rel = next((i for i, w in enumerate(words) if w in RELATIVE), None)
        lits = []
        rel_lits = []
        if rel is not None:
            rc = toks[rel + 1 :]
            clauses = self._conjunction(rc, ctx, Clause(x, []), subject=x)
            for c in clauses:
                if c.kind != "conj":
                    raise ParseFailure("disjunction in relative clause")
                rel_lits.extend(c.lits)
            toks = toks[:rel]
            words = words[:rel]
        # Post-nominal modifiers: "people with red hair" -> skip; "dogs in the park"
        if any(w in PREPOSITIONS for w in words[1:]):
            k = next(i for i, w in enumerate(words) if i > 0 and w in PREPOSITIONS)
            pp = toks[k:]
            toks, words = toks[:k], words[:k]
            pred = "_".join(t.text for t in pp if t.text not in DETERMINERS and t.text != ",")
            if not toks:
                raise ParseFailure("empty noun phrase")
            lits.append((True, f"{singular(words[-1])}_{pred}", (x,)))
        words = [w for w in words if w not in (",", "and", "a", "an", "the")]
        if not words:
            raise ParseFailure("empty restrictor")
        for w in words[:-1]:
            if w in COPULA | AUX | MODALS or w in RELATIVE:
                raise ParseFailure(f"unexpected {w!r} in noun phrase")
            if w == "not":
                raise ParseFailure("negation in noun phrase")
            lits.append((True, w, (x,)))
        head = words[-1]
        if head in COPULA | AUX | MODALS:
            raise ParseFailure("verb in noun phrase")
        if head not in GENERIC_NOUNS:
            lits.append((True, singular(head), (x,)))
        return lits + rel_lits

    # -- clause level -------------------------------------------------------------

    def _starts_with_entity(self, toks):
        if not toks:
            return False
        w = toks[0].text
        if w == "the" or w in PRONOUNS:
            return True
        if w in self.known_entities:
            return True
        if toks[0].cap:
            # Sentence-initial capitals are ambiguous ("Cold people are ...").
            k = 1
            while k < len(toks) and toks[k].cap:
                k += 1
            if k + 1 < len(toks) and toks[k].text in ("and", ",") and (
                toks[k + 1].cap or toks[k + 1].text == "the"
            ):
                return True  # coordinated subject: "Bob and Anne ..."
            return k < len(toks) and self._starts_vp(toks[k:])
        # Lower-case proper names: accept if directly followed by a verb.
        if len(toks) > 1 and (toks[1].text in COPULA | AUX | MODALS) and not looks_plural(w):
            return True
        return False

    def _conjunction(self, toks, ctx, prev, head=False, subject=None):
        """Coordinated clauses with subject/verb ellipsis."""
        parts = _split_coordination(toks)
        clauses = []
        pending_subjects = []  # coordinated subjects: "the cat and the dog are big"
        for part in parts:
            if not part:
                continue
            if not clauses and subject is None:
                np = self._subject(part, ctx)
                if np is not None and not np[1]:
                    pending_subjects.append(np[0])
                    continue
                if pending_subjects and part[0].cap and (np is None or not np[1]):
                    # Plural verb after coordinated names: "Bob and Anne like ..."
                    k = 1
                    while k < len(part) and part[k].cap:
                        k += 1
                    np = (entity_name([t.text for t in part[:k]]), part[k:])
                if pending_subjects and np is not None and np[1]:
                    for s in [*pending_subjects, np[0]]:
                        clauses.append(self._verb_phrase(np[1], s, ctx))
                    prev = clauses[-1]
                    pending_subjects = []
                    continue
            c = self._clause(part, ctx, prev, subject)
            clauses.append(c)
            prev = c
            subject = None
        if pending_subjects:
            raise ParseFailure("coordinated noun phrase without verb phrase")
        if not clauses:
            raise ParseFailure("empty clause")
        return clauses

    def _clause(self, toks, ctx, prev, forced_subject=None):
        words = [t.text for t in toks]
        if forced_subject is not None:
            subj, rest = forced_subject, toks
        else:
            parsed = self._subject_with_vp(toks, ctx)
            if parsed is None:
                if prev is None:
                    raise ParseFailure(f"no subject in {' '.join(words)!r}")
                subj = prev.subject
                rest = toks
                if prev.copular and words[0] in ("not", "never") and (
                    len(words) == 1 or words[1] not in COPULA | AUX | MODALS
                ):
                    rest = [Tok("is", False)] + rest
                elif prev.indefinite and not self._starts_vp(rest):
                    # "is a cold, round man": adjectives continue the noun phrase.
                    rest = [Tok("is", False), Tok("a", False)] + rest
                elif not self._starts_vp(rest):
                    # Ellipsis: "is red and (is) big", "likes the cat and (likes) the dog"
                    if prev.copular:
                        rest = [Tok("is", False)] + ([Tok("not", False)] if prev.negated
                                                     and words[0] not in NEGATORS else []) + rest
                    elif prev.verb:
                        rest = [Tok(prev.verb, False)] + rest
                    else:
                        rest = [Tok("is", False)] + rest
            else:
                subj, rest = parsed
        return self._verb_phrase(rest, subj, ctx)

    def _known_lemmas(self):
        if len(self._lemma_cache[0]) != len(self.known_verbs):
            self._lemma_cache = (set(self.known_verbs), {lemma_verb(v) for v in self.known_verbs})
        return self._lemma_cache[1]

    def _subject_with_vp(self, toks, ctx):
        parsed = self._subject(toks, ctx)
        if parsed is not None and not parsed[1]:
            return None  # a bare noun phrase is an elided object, not a subject
        return parsed

    def _starts_vp(self, toks):
        if not toks:
            return False
        w = toks[0].text
        return w in COPULA | AUX | MODALS | {"has", "have", "not", "never"} or (
            self._is_verb_word(w) and (len(toks) > 1 or w in self.known_verbs)
        )

    def _is_verb_word(self, w):
        if w in self.known_verbs or w in self._known_lemmas():
            return True
        return (w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")) and len(w) > 2
                and w not in GENERIC_NOUNS and w not in ("this", "has", "was", "its", "his",
                                                         "hers", "theirs", "yes", "less"))

    def _subject(self, toks, ctx):
        """Return (term, remaining tokens) or None if the clause has no subject."""
        w = toks[0].text
        if w in PRONOUNS:
            if ctx.main is not None:
                return ctx.main, toks[1:]
            if ctx.last_entity is not None:
                return ctx.last_entity, toks[1:]
            x = ctx.variable("x")
            ctx.main = x
            return x, toks[1:]
        if w in INDEFINITES:
            if ctx.main is None:
                ctx.main = ctx.variable("x")
            return ctx.main, toks[1:]
        if w in ("a", "an") and len(toks) > 2:
            k = self._np_end(toks, 1)
            if k is not None:
                if ctx.main is None:
                    ctx.main = ctx.variable("x")
                return ctx.main, toks[k:]
        if w == "the":
            k = self._np_end(toks, 1)
            if k is None:
                return None
            name = entity_name([t.text for t in toks[1:k] if t.text != ","])
            self.known_entities.add(name)
            ctx.last_entity = name
            return name, toks[k:]
        if toks[0].cap or w in self.known_entities:
            k = 1
            while k < len(toks) and toks[k].cap and toks[k].text not in COPULA:
                k += 1
            if k == len(toks) or self._starts_vp(toks[k:]):
                name = entity_name([t.text for t in toks[:k]])
                self.known_entities.add(name)
                ctx.last_entity = name
                return name, toks[k:]
            # fall through: maybe a multi-word lower-case name
        # Lower-case name directly followed by a verb: "bob is big".
        if len(toks) > 1 and toks[1].text in COPULA | AUX | MODALS and w not in NEGATORS:
            if w in ("and", "or", ",", "then") or w in PREPOSITIONS:
                return None
            if ctx.main is not None and w in ("someone", "something"):
                return ctx.main, toks[1:]
            if w not in self.known_entities and not toks[0].cap and len(toks) > 2:
                # Could be an adjective in an elided clause ("red is"?); accept as name.
                pass
            self.known_entities.add(w)
            return w, toks[1:]
        return None

    def _np_end(self, toks, start):
        """End index of a noun phrase starting at ``start`` (exclusive)."""
        for k in range(start + 1, len(toks) + 1):
            if k == len(toks):
                return k
            if self._starts_vp(toks[k:]) and toks[k].text not in ("not", "never"):
                return k
            if toks[k].text in RELATIVE:
                return None
        return None

    def _object(self, toks, ctx):
        """Parse an object noun phrase. Returns (term or None, unary preds)."""
        words = [t.text for t in toks if t.text != ","]
        if not words:
            return None
        if len(words) == 1 and (words[0] in PRONOUNS):
            if words[0] in ("itself", "themselves", "himself", "herself"):
                return ctx.main, []
            if ctx.main is not None:
                return ctx.main, []
            return ctx.last_entity, []
        if len(words) == 1 and words[0] in INDEFINITES:
            if ctx.main is None:
                ctx.main = ctx.variable("x")
                return ctx.main, []
            y = ctx.variable(f"y{len(ctx.vars_)}")
            return y, []
        if words[0] == "the":
            name = entity_name(words[1:])
            self.known_entities.add(name)
            return name, []
        if toks[0].cap or words[0] in self.known_entities or len(words) == 1:
            name = entity_name(words)
            self.known_entities.add(name)
            return name, []
        return None

    def _verb_phrase(self, toks, subj, ctx) -> Clause:
        words = [t.text for t in toks]
        if not words:
            raise ParseFailure("missing verb phrase")
        neg = False
        i = 0
        w = words[0]
        if w in MODALS:
            i += 1
            if i < len(words) and words[i] in NEGATORS:
                neg = True
                i += 1
            if i < len(words) and words[i] == "be":
                words = ["is"] + words[i + 1 :]
                toks = [Tok("is", False)] + toks[i + 1 :]
                i = 0
                w = "is"
            else:
                rest = toks[i:]
                c = self._verb_phrase(rest, subj, ctx)
                if neg:
                    c.lits = [(not p, q, a) for p, q, a in c.lits]
                    c.negated = True
                return c
        if w in COPULA:
            i = 1
            if i < len(words) and words[i] in NEGATORS and words[i] != "no":
                neg = True
                i += 1
            comp = toks[i:]
            return self._complement(comp, subj, neg, ctx)
        if w in AUX:
            i = 1
            if i < len(words) and words[i] in NEGATORS:
                neg = True
                i += 1
            rest = toks[i:]
            if not rest:
                raise ParseFailure("dangling auxiliary")
            c = self._verb_phrase(rest, subj, ctx)
            if neg:
                c.lits = [(not p, q, a) for p, q, a in c.lits]
                c.negated = True
            return c
        if w in ("not", "never"):
            c = self._verb_phrase(toks[1:], subj, ctx)
            c.lits = [(not p, q, a) for p, q, a in c.lits]
            c.negated = True
            return c
        # Lexical verb.
        verb = lemma_verb(w)
        self._pending_verbs.add(w)
        rest = toks[1:]
        if not rest:
            return Clause(subj, [(True, verb, (subj,))], verb=w)
        rwords = [t.text for t in rest]
        if rwords[0] in PREPOSITIONS and len(rwords) > 1:
            verb = f"{verb}_{rwords[0]}"
            rest = rest[1:]
            rwords = rwords[1:]
        if rwords[0] in ("a", "an") or (rwords[0] not in ("the",) and not rest[0].cap
                                        and rwords[0] not in PRONOUNS | INDEFINITES
                                        and rwords[0] not in self.known_entities
                                        and len(rwords) > 1):
            # "has a tail", "plays tennis well" -> unary predicate
            content = [x for x in rwords if x not in ("a", "an", ",")]
            if any(x in COPULA | AUX | MODALS | RELATIVE for x in content):
                raise ParseFailure("clause inside verb object")
            pred = f"{verb}_" + "_".join(singular(x) for x in content)
            return Clause(subj, [(True, pred, (subj,))], verb=w)
        obj = self._object(rest, ctx)
        if obj is None:
            raise ParseFailure(f"cannot parse object {' '.join(rwords)!r}")
        term, _ = obj
        return Clause(subj, [(True, verb, (subj, term))], verb=w)

    def _complement(self, toks, subj, neg, ctx) -> Clause:
        words = [t.text for t in toks]
        if not words:
            raise ParseFailure("empty complement")
        if words[0] == "neither" and "nor" in words:
            k = words.index("nor")
            a = self._complement(toks[1:k], subj, False, ctx)
            b = self._complement(toks[k + 1 :], subj, False, ctx)
            lits = [(not p, q, ar) for p, q, ar in a.lits + b.lits]
            return Clause(subj, lits, copular=True)
        if words[0] == "either" and "or" in words:
            k = words.index("or")
            a = self._complement(toks[1:k], subj, False, ctx)
            b = self._complement(toks[k + 1 :], subj, False, ctx)
            if len(a.lits) == 1 and len(b.lits) == 1:
                return Clause(subj, a.lits + b.lits, kind="or", copular=True)
            raise ParseFailure("complex either/or")
        if "or" in words and words.index("or") > 0:
            k = words.index("or")
            a = self._complement(toks[:k], subj, neg, ctx)
            b = self._complement(toks[k + 1 :], subj, neg, ctx)
            if len(a.lits) == 1 and len(b.lits) == 1 and not neg:
                return Clause(subj, a.lits + b.lits, kind="or", copular=True)
            raise ParseFailure("complex disjunctive complement")
        if words[0] in ("a", "an"):
            content = [t for t in toks[1:] if t.text not in (",", "and")]
            if not content:
                raise ParseFailure("empty noun complement")
            rel = next((i for i, t in enumerate(content) if t.text in RELATIVE), None)
            extra = []
            if rel is not None:
                rc = self._conjunction(content[rel + 1 :], ctx, Clause(subj, []), subject=subj)
                for c in rc:
                    extra.extend(c.lits)
                content = content[:rel]
            pp = next((i for i, t in enumerate(content) if i > 0 and t.text in PREPOSITIONS), None)
            if pp is not None:
                obj = self._object(content[pp + 1 :], ctx) if pp + 1 < len(content) else None
                noun = singular(content[pp - 1].text)
                if obj is not None:
                    extra.append((True, f"{noun}_{content[pp].text}", (subj, obj[0])))
                content = content[:pp]
            lits = [(True, t.text, (subj,)) for t in content[:-1]]
            head = content[-1].text
            if head not in GENERIC_NOUNS or not lits:
                lits.append((True, singular(head), (subj,)))
            lits.extend(extra)
            if neg:
                if len(lits) != 1:
                    # ¬(a ∧ b): keep only the noun (the most specific claim).
                    lits = [lits[-1]] if not extra else lits[:1]
                lits = [(False, q, a) for _, q, a in lits]
            return Clause(subj, lits, copular=True, negated=neg, indefinite=True)
        if words[0] == "the" or (toks[0].cap and len(words) <= 3):
            obj = self._object(toks, ctx)
            if obj is not None:
                return Clause(subj, [(not neg, "=", (subj, obj[0]))], copular=True, negated=neg)
        # Adjective, optionally followed by a prepositional object: "afraid of the dog".
        content = [w for w in words if w != ","]
        if len(content) == 1:
            pred = singular(content[0]) if looks_plural(content[0]) else content[0]
            return Clause(subj, [(not neg, pred, (subj,))], copular=True, negated=neg)
        if len(content) >= 2 and content[1] in PREPOSITIONS:
            obj = self._object(toks[2:], ctx)
            if obj is not None:
                pred = f"{content[0]}_{content[1]}"
                return Clause(subj, [(not neg, pred, (subj, obj[0]))], copular=True, negated=neg)
        if all(w not in COPULA | AUX | DETERMINERS for w in content) and len(content) <= 4:
            # Multi-word adjective / plural noun complement ("are big dogs", "is well known").
            if looks_plural(content[-1]) and len(content) > 1:
                lits = [(not neg, w, (subj,)) for w in content[:-1]]
                lits.append((not neg, singular(content[-1]), (subj,)))
                return Clause(subj, lits, copular=True, negated=neg)
            return Clause(subj, [(not neg, "_".join(content), (subj,))], copular=True, negated=neg)
        raise ParseFailure(f"cannot parse complement {' '.join(words)!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    text = text.replace("\n", " . ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip(" .")]


def _split_top(toks, word):
    groups, cur = [], []
    for t in toks:
        if t.text == word:
            if cur and cur[-1].text == ",":
                cur = cur[:-1]
            groups.append(cur)
            cur = []
        else:
            cur.append(t)
    groups.append(cur)
    return [g for g in groups if g]


def _split_coordination(toks):
    """Split on 'and' / ',' outside either..or / neither..nor complements."""
    words = [t.text for t in toks]
    if "either" in words or "neither" in words:
        # Only split on 'and' preceding the either/neither construction.
        k = words.index("either") if "either" in words else words.index("neither")
        head = _split_coordination(toks[:k]) if k > 0 else [[]]
        if head and head[-1] is not None:
            head[-1] = head[-1] + toks[k:]
        return head
    parts, cur = [], []
    for t in toks:
        if t.text in ("and", ",", "but"):
            if cur:
                parts.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        parts.append(cur)
    return parts


def _lit_formula(lit):
    positive, pred, args = lit
    atom = ("atom", pred, args)
    return atom if positive else ("not", atom)


def _vars_in_lits(lits):
    out = []
    for _, _, args in lits:
        for a in args:
            if type(a) is Var and a not in out:
                out.append(a)
    return out


def _implication(body_lits, head_clause):
    body = ("and", *(_lit_formula(lit) for lit in body_lits)) if body_lits else ("true",)
    head = ("or", *(_lit_formula(lit) for lit in head_clause.lits))
    f = ("imp", body, head)
    for v in _vars_in_lits(body_lits + head_clause.lits):
        f = ("all", v, f)
    return f


def _negated_universal(restrictor, clause):
    body = ("and", *(_lit_formula(lit) for lit in restrictor)) if restrictor else ("true",)
    inner = ("and", *(_lit_formula(lit) for lit in clause.lits))
    if clause.kind == "or":
        inner = ("or", *(_lit_formula(lit) for lit in clause.lits))
    f = ("imp", body, ("not", inner))
    for v in _vars_in_lits(restrictor + clause.lits):
        f = ("all", v, f)
    return f
