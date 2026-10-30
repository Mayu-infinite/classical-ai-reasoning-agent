import pytest

from reasoning_agent.english import EnglishParser, lemma_verb, singular


def parse(sentence):
    return sorted(str(item) for item in EnglishParser().parse_sentence(sentence))


def test_lemmatization():
    assert lemma_verb("chases") == lemma_verb("chase") == "chase"
    assert lemma_verb("watches") == "watch"
    assert singular("people") == "person"


@pytest.mark.parametrize("sentence, expected", [
    ("Bob is young.", ["young(bob)"]),
    ("The bald eagle does not chase the rabbit.", ["¬chase(bald_eagle, rabbit)"]),
    ("Anne likes the cat and the dog.", ["like(anne, cat)", "like(anne, dog)"]),
    ("The cat and the dog are big.", ["big(cat)", "big(dog)"]),
    ("Fiona is nice, kind, and young.", ["kind(fiona)", "nice(fiona)", "young(fiona)"]),
    ("Dave is a cold, round man.", ["cold(dave)", "round(dave)"]),
    ("Alan, who is round and red, tends to be rather blue.",
     ["blue(alan)", "red(alan)", "round(alan)"]),
])
def test_facts(sentence, expected):
    assert parse(sentence) == expected


@pytest.mark.parametrize("sentence, body, head", [
    ("If someone is young then they are smart.", ["young"], "smart"),
    ("If someone is green and round, they tend to be young.", ["green", "round"], "young"),
    ("All smart, white people are big.", ["smart", "white"], "big"),
    ("Kind, young things are not smart.", ["kind", "young"], "¬smart"),
    ("People who are young and big are also round.", ["young", "big"], "round"),
    ("A big round young person is often red.", ["big", "round", "young"], "red"),
    ("Big people are red if they are young.", ["big", "young"], "red"),
    ("Everyone who is young is kind.", ["young"], "kind"),
    ("If something is not big then it is kind.", ["¬big"], "kind"),
])
def test_rules(sentence, body, head):
    (rule,) = EnglishParser().parse_sentence(sentence)
    got_body = [("" if p else "¬") + q for p, q, _ in rule.body]
    assert got_body == body
    assert ("" if rule.head[0] else "¬") + rule.head[1] == head
    variables = {a for _, _, args in (*rule.body, rule.head) for a in args}
    assert len(variables) == 1


def test_rule_mixing_variables_and_constants():
    (rule,) = EnglishParser().parse_sentence(
        "If something chases the dog and the dog is big then it is kind.")
    assert [q for _, q, _ in rule.body] == ["chase", "big"]
    assert rule.body[1][2] == ("dog",)


def test_disjunction_becomes_formula():
    items = EnglishParser().parse_sentence("James is either a teacher or a doctor.")
    assert items[0][0] == "formula"


def test_controlled_language_detection():
    parser = EnglishParser()
    assert parser.is_controlled(parser.parse_sentence("If someone is red then they are big."))
    assert not parser.is_controlled(parser.parse_sentence("Bonnie is in this club."))
