from reasoning_agent.datalog import BackwardChainer, ForwardChainer, Rule
from reasoning_agent.logic import Var


def lit(pred, *args, positive=True):
    return (positive, pred, tuple(args))


def program():
    x = Var("x")
    return [
        Rule((), lit("young", "bob")),
        Rule((), lit("kind", "bob")),
        Rule((), lit("big", "anne")),
        Rule((lit("young", x),), lit("smart", x)),
        Rule((lit("smart", x), lit("kind", x)), lit("successful", x)),
        Rule((lit("big", x, positive=False),), lit("small", x)),
    ]


def test_forward_chaining_derives_closure():
    model = ForwardChainer(program()).model()
    assert model.holds(True, "successful", ("bob",))
    assert not model.holds(True, "successful", ("anne",))


def test_negation_as_failure():
    model = ForwardChainer(program()).model()
    assert model.holds(True, "small", ("bob",))
    assert not model.holds(True, "small", ("anne",))
    assert model.holds(False, "smart", ("anne",))


def test_backward_chaining_agrees_with_forward_chaining():
    rules = program()
    model = ForwardChainer(rules).model()
    bc = BackwardChainer(rules)
    for atom in [("successful", ("bob",)), ("small", ("anne",)), ("smart", ("anne",)),
                 ("small", ("bob",))]:
        assert bc.prove((True, *atom)) == model.holds(True, *atom)


def test_well_founded_semantics_handles_unstratified_loop():
    # p :- not q.  q :- not p.  -> both undefined, neither is true.
    rules = [Rule((lit("q", "a", positive=False),), lit("p", "a")),
             Rule((lit("p", "a", positive=False),), lit("q", "a"))]
    model = ForwardChainer(rules).model()
    assert not model.holds(True, "p", ("a",))
    assert not model.holds(False, "p", ("a",))


def test_relations_with_two_variables():
    x, y = Var("x"), Var("y")
    rules = [Rule((), lit("chase", "cat", "mouse")),
             Rule((lit("chase", x, y),), lit("afraid", y, x))]
    model = ForwardChainer(rules).model()
    assert model.holds(True, "afraid", ("mouse", "cat"))
