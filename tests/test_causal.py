from reasoning_agent.causal import answer_causal, parse_causal_context, parse_causal_query


def test_parse_mixed_causal_statements():
    model, obs = parse_causal_context(
        "We know that A causes not B, B and C together cause D, D or A causes E, "
        "and E causes F and G. We observed C.")
    assert model.equations["B"] == ("not", ("var", "A"))
    assert model.equations["D"][0] == "and"
    assert model.equations["E"][0] == "or"
    assert "F" in model.equations and "G" in model.equations
    assert obs == {"C": True}


def test_query_forms():
    q = parse_causal_query("Would Y occur if not X instead of X?")
    assert q.interventions == {"X": False} and q.factual == {"X": True}
    q = parse_causal_query("Would Y occur if not X and not Z?")
    assert q.interventions == {"X": False, "Z": False}
    q = parse_causal_query("Assume not X, and based on this assumption, further suppose not Z. "
                           "Would Y occur?")
    assert q.interventions == {"X": False, "Z": False}


def test_basic_counterfactual():
    ctx = ("We know that Ziklo causes not Blaf, Blaf causes Trune, Trune or Ziklo causes Vork, "
           "Vork causes Sline, and Sline causes Lumbo.")
    assert answer_causal(ctx, "Would Lumbo occur if not Ziklo instead of Ziklo?").answer is True


def test_conditional_counterfactual_uses_observed_roots():
    ctx = ("We know that Fizo and Blorn together cause Plim, Plim causes Quaz, and Quaz causes "
           "Jext. We observed Blorn.")
    assert answer_causal(ctx, "Would Jext occur if Fizo instead of not Fizo?").answer is True
    assert answer_causal(ctx, "Would Jext occur if not Fizo instead of Fizo?").answer is False


def test_backdoor_abduction():
    ctx = ("We know that Blorn causes Fizo, Fizo or Blorn causes Plim, Plim causes Quaz, and "
           "Quaz causes Jext. Blorn~ Bern(0.2). We observed Plim")
    # Fizo occurred, so Blorn must have occurred; Blorn alone keeps Plim on.
    assert answer_causal(ctx, "Would Jext occur if not Fizo instead of Fizo?").answer is True
