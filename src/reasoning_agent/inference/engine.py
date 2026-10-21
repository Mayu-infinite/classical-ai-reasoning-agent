from reasoning_agent.inference.forward_chaining import forward_chain
from reasoning_agent.logic.ast import Atom
from reasoning_agent.logic.knowledge_base import KnowledgeBase


class ReasoningEngine:
    """Reason over finite positive Horn knowledge bases."""

    def __init__(self, max_facts: int = 100_000) -> None:
        self.max_facts = max_facts

    def entails(
        self,
        knowledge_base: KnowledgeBase,
        query: Atom,
    ) -> bool:
        if any(term.is_variable for term in query.arguments):
            raise ValueError("The initial engine requires a ground query")

        derived_facts = forward_chain(
            knowledge_base,
            max_facts=self.max_facts,
        )

        return query in derived_facts
