from reasoning_agent.inference.engine import ReasoningEngine
from reasoning_agent.logic.ast import Atom, Term
from reasoning_agent.logic.knowledge_base import HornRule, KnowledgeBase


def main() -> None:
    bob = Term("Bob")
    x = Term("x", is_variable=True)

    knowledge_base = KnowledgeBase(
        facts={Atom("Young", (bob,))},
        rules=[
            HornRule(
                premises=(Atom("Young", (x,)),),
                conclusion=Atom("Smart", (x,)),
            )
        ],
    )

    engine = ReasoningEngine()
    query = Atom("Smart", (bob,))

    print("Smoke-test prediction:", engine.entails(knowledge_base, query))


if __name__ == "__main__":
    main()
