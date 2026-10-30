
"""A restricted natural-language adapter for RuleTaker.

Supports:
- Entity properties: "The bald eagle is big."
- Negative properties: "Anne is not young."
- Binary relations: "The bald eagle eats the bear."
- Property rules: "If someone is kind then they are red."
- Conjunctive rules with positive or negated premises.
- Rules such as "Kind, young things are not smart."
- Rules such as "All green, red people are not young."

Negated properties are represented as explicit predicates, such as
NotYoung(Anne). This adapter is not a full natural-language parser,
and explicit negative predicates do not implement full negation semantics.
"""

import re
from typing import Any

WORD = r"[A-Za-z]+(?:[-'][A-Za-z]+)*"
ENTITY = rf"{WORD}(?:\s+{WORD})*"

# Relations found in the inspected RuleTaker examples.
# Add other relation verbs here as needed.
RELATIONS = {
    "eats",
    "likes",
    "chases",
    "sees",
    "needs",
    "visits",
    "likes",
    "loves",
    "hates",
    "helps",
    "supports",
    "admires",
    "knows",
    "follows",
    "runs",
    "moves",
    "believes",
    "wants",
    "likes",
    "attacks",
    "protects",
    "respects",
    "trusts",
    "calls",
    "finds",
    "helps",
    "avoids",
    "contains",
    "requires",
    "wears",
    "holds",
    "takes",
    "gives",
    "makes",
    "plays",
    "visits",
}

FACT_PATTERN = re.compile(
    rf"^(?:the\s+)?(?P<subject>{ENTITY})\s+is\s+"
    rf"(?P<negation>not\s+)?(?P<property>{WORD})$",
    re.IGNORECASE,
)

IF_RULE_PATTERN = re.compile(
    rf"^if\s+someone\s+is\s+(?P<premises>.+?)\s+"
    rf"then\s+they\s+are\s+(?P<negation>not\s+)?"
    rf"(?P<conclusion>{WORD})$",
    re.IGNORECASE,
)

THINGS_RULE_PATTERN = re.compile(
    rf"^(?:all\s+)?(?P<premises>{WORD}"
    rf"(?:\s*,\s*{WORD})*)\s+"
    rf"(?:things|people)\s+are\s+"
    rf"(?P<negation>not\s+)?(?P<conclusion>{WORD})$",
    re.IGNORECASE,
)


def canonical_name(text: str) -> str:
    """Convert a name or word to a parser-compatible identifier."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        raise ValueError(f"Invalid name: {text!r}")
    return "".join(word[0].upper() + word[1:].lower() for word in words)


def predicate_name(word: str, negated: bool = False) -> str:
    """Represent a negated property using an explicit predicate."""
    name = canonical_name(word)
    return f"Not{name}" if negated else name


def _parse_rule(sentence: str) -> str | None:
    """Parse a supported rule into a Horn-style rule string."""
    match = IF_RULE_PATTERN.fullmatch(sentence)

    if match:
        premises_text = match.group("premises").strip()
        premise_parts = re.split(
            r"\s+and\s+", premises_text, flags=re.IGNORECASE
        )
        premise_literals: list[str] = []

        for part in premise_parts:
            literal_match = re.fullmatch(
                rf"\s*(not\s+)?({WORD})\s*",
                part,
                re.IGNORECASE,
            )
            if literal_match is None:
                return None

            negated, property_name = literal_match.groups()
            predicate = predicate_name(
                property_name,
                negated=bool(negated),
            )
            premise_literals.append(f"{predicate}(x)")

        conclusion = predicate_name(
            match.group("conclusion"),
            negated=bool(match.group("negation")),
        )
        return f"{' & '.join(premise_literals)} -> {conclusion}(x)"

    match = THINGS_RULE_PATTERN.fullmatch(sentence)

    if match:
        premise_parts = re.split(
            r"\s*,\s*", match.group("premises")
        )
        premises = [
            f"{predicate_name(part)}(x)"
            for part in premise_parts
        ]
        conclusion = predicate_name(
            match.group("conclusion"),
            negated=bool(match.group("negation")),
        )
        return f"{' & '.join(premises)} -> {conclusion}(x)"

    return None


def _parse_relation(sentence: str) -> str | None:
    """Parse a binary relation using known relation verbs."""
    words = sentence.split()

    # Try each known relation, longest subject first.
    for index, word in enumerate(words):
        relation_word = word.lower().strip(".,!?")

        if relation_word not in RELATIONS:
            continue

        subject_words = words[:index]
        object_words = words[index + 1 :]

        if subject_words and subject_words[0].lower() == "the":
            subject_words = subject_words[1:]

        if object_words and object_words[0].lower() == "the":
            object_words = object_words[1:]

        subject_text = " ".join(subject_words).strip()
        object_text = " ".join(object_words).strip()

        if not subject_text or not object_text:
            continue

        subject = canonical_name(subject_text)
        relation = canonical_name(relation_word)
        obj = canonical_name(object_text)

        return f"{relation}({subject},{obj})"

    return None


def parse_property_sentence(
    sentence: str,
) -> tuple[str, str] | None:
    """Return ('fact' or 'rule', logic string), or None if unsupported."""
    sentence = sentence.strip().rstrip(".").strip()

    if not sentence:
        return None

    rule = _parse_rule(sentence)
    if rule is not None:
        return "rule", rule

    match = FACT_PATTERN.fullmatch(sentence)
    if match:
        subject = canonical_name(match.group("subject"))
        predicate = predicate_name(
            match.group("property"),
            negated=bool(match.group("negation")),
        )
        return "fact", f"{predicate}({subject})"

    relation = _parse_relation(sentence)
    if relation is not None:
        return "fact", relation

    return None


def adapt_ruletaker_record(
    record: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert a supported RuleTaker record to the engine's input format."""
    context = record.get("context")
    question = record.get("question")
    raw_label = str(record.get("label", "")).strip().lower()

    if not isinstance(context, str) or not isinstance(question, str):
        return None, "Missing text context or question"

    if raw_label not in {"entailment", "not entailment"}:
        return None, f"Unsupported label: {raw_label!r}"

    facts: list[str] = []
    rules: list[str] = []

    sentences = [
        part.strip()
        for part in context.split(".")
        if part.strip()
    ]

    for sentence in sentences:
        parsed = parse_property_sentence(sentence)

        if parsed is None:
            return None, f"Unsupported context sentence: {sentence!r}"

        kind, logic = parsed

        if kind == "fact":
            facts.append(logic)
        else:
            rules.append(logic)

    parsed_query = parse_property_sentence(question)

    if parsed_query is None or parsed_query[0] != "fact":
        return None, f"Unsupported question: {question!r}"

    return {
        "id": str(record.get("example_id", record.get("question_id", ""))),
        "facts": facts,
        "rules": rules,
        "query": parsed_query[1],
        "label": raw_label == "entailment",
    }, None
