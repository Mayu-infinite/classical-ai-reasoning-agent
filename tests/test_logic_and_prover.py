from reasoning_agent.logic import Var, parse_fol, repair_symbols, to_clauses
from reasoning_agent.prover import Prover, prove_formula, subsumes, unify


def proves(premises, goal, **kw):
    return prove_formula([parse_fol(p) for p in premises], parse_fol(goal), Prover(**kw)).proved


def test_parse_quantifiers_and_connectives():
    f = parse_fol("∀x (Dog(x) ∧ ¬Small(x) → Animal(x) ⊕ Robot(x))")
    assert f[0] == "all"
    assert f[2][0] == "imp"


def test_parser_repairs_unbalanced_parentheses():
    f = parse_fol("InterestedIn(erica, puzzle) ∧ Creative(erica)) → ¬Creative(erica)")
    assert f[0] == "imp"


def test_free_variables_are_universal():
    f = parse_fol("P(x) → Q(x)")
    assert f[0] == "all"


def test_free_variables_of_a_goal_are_existential():
    goal = parse_fol("Cost(gre, x) ∧ Cheaper(x, 300)", free_quantifier="ex")
    premises = [parse_fol("Cost(gre, 205)"), parse_fol("Cheaper(205, 300)")]
    assert prove_formula(premises, goal, Prover()).proved


def test_quantifier_without_variable_binds_free_variables():
    f = parse_fol("∃(CarBody(x) ∧ Made(x, colli))")
    assert f[0] == "ex"


def test_tokenizer_variants():
    assert parse_fol("LostTo(coco, świątek)")[2] == ("coco", "świątek")
    assert parse_fol("P(bob) —> Q(bob)")[0] == "imp"
    assert parse_fol("P(bob) v (Q(bob))")[0] == "or"


def test_unification_and_occurs_check():
    x, y = Var("x"), Var("y")
    assert unify(("f", x), ("f", "a"), {}) == {x: "a"}
    assert unify(x, ("f", x), {}) is None
    assert unify(("f", x, "b"), ("f", "a", y), {}) == {x: "a", y: "b"}


def test_subsumption():
    x = Var("x")
    assert subsumes(((True, "P", (x,)),), ((True, "P", ("a",)), (False, "Q", ("a",))))
    assert not subsumes(((True, "P", ("b",)),), ((True, "P", ("a",)),))


def test_modus_ponens_chain():
    premises = ["∀x (Young(x) → Smart(x))", "∀x (Smart(x) ∧ Kind(x) → Successful(x))",
                "Young(bob)", "Kind(bob)"]
    assert proves(premises, "Successful(bob)")
    assert not proves(premises, "Successful(alice)")


def test_reasoning_by_cases_with_xor():
    premises = ["A(c) ⊕ B(c)", "A(c) → D(c)", "B(c) → D(c)"]
    assert proves(premises, "D(c)")


def test_existential_and_skolemization():
    premises = ["∀x (Man(x) → Mortal(x))", "∃x Man(x)"]
    assert proves(premises, "∃y Mortal(y)")
    assert not proves(premises, "∀y Mortal(y)")


def test_equality_reasoning():
    assert proves(["a = b", "P(a)"], "P(b)")


def test_tseitin_definitions_keep_cnf_small():
    big = " ∨ ".join(f"(A{i}(c) ∧ B{i}(c) ∧ C{i}(c))" for i in range(8))
    clauses = to_clauses(parse_fol(big))
    assert len(clauses) < 100


def test_symbol_repair_merges_typos():
    premises = [parse_fol("∀x (GrewUpIn(x, home) → Eat(x, salad))"),
                parse_fol("GrewUpIn(taylor, home)")]
    goal = parse_fol("¬GrowUpIn(taylor, home)")
    _, _, changes = repair_symbols(premises, goal)
    assert changes.get("GrowUpIn/2") == "GrewUpIn"


def test_symbol_repair_maps_goal_only_symbols():
    premises = [parse_fol("LargestPopulation(central)")]
    goal = parse_fol("LargestPopulationIn(central)")
    _, _, changes = repair_symbols(premises, goal)
    assert changes.get("LargestPopulationIn/1") == "LargestPopulation"
