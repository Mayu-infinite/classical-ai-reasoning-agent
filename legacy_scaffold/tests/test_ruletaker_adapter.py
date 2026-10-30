
def test_multiword_entities_and_relations():
    from reasoning_agent.data.ruletaker import adapt_ruletaker_record

    record = {
        "context": (
            "The bald eagle eats the bear. "
            "The bald eagle is big."
        ),
        "question": "The bald eagle eats the bear.",
        "label": "entailment",
    }

    adapted, reason = adapt_ruletaker_record(record)

    assert reason is None
    assert adapted is not None
    assert "Eats(BaldEagle,Bear)" in adapted["facts"]
    assert "Big(BaldEagle)" in adapted["facts"]
    assert adapted["query"] == "Eats(BaldEagle,Bear)"


def test_if_then_property_rule():
    from reasoning_agent.data.ruletaker import adapt_ruletaker_record

    record = {
        "context": (
            "The cat is kind. "
            "If someone is kind then they are red."
        ),
        "question": "The cat is red.",
        "label": "entailment",
    }

    adapted, reason = adapt_ruletaker_record(record)

    assert reason is None
    assert adapted is not None
    assert "Kind(Cat)" in adapted["facts"]
    assert "Kind(x) -> Red(x)" in adapted["rules"]


def test_conjunctive_property_rule():
    from reasoning_agent.data.ruletaker import adapt_ruletaker_record

    record = {
        "context": (
            "The cat is nice. "
            "The cat is rough. "
            "The cat is cold. "
            "If someone is nice and rough and cold then they are big."
        ),
        "question": "The cat is big.",
        "label": "entailment",
    }

    adapted, reason = adapt_ruletaker_record(record)

    assert reason is None
    assert adapted is not None
    assert (
        "Nice(x) & Rough(x) & Cold(x) -> Big(x)"
        in adapted["rules"]
    )
