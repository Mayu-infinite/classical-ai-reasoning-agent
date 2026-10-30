
from reasoning_agent.inference.backward_chaining import backward_chain
from reasoning_agent.inference.forward_chaining import forward_chain
from reasoning_agent.logic.ast import Atom
from reasoning_agent.logic.knowledge_base import KnowledgeBase


class ReasoningEngine:
    """Reason over finite positive Horn knowledge bases."""

    def __init__(
        self,
        max_facts: int = 100_000,
        max_depth: int = 100,
    ) -> None:
        if max_facts < 0:
            raise ValueError("max_facts cannot be negative")
        if max_depth < 0:
            raise ValueError("max_depth cannot be negative")

        self.max_facts = max_facts
        self.max_depth = max_depth

    def entails(
        self,
        knowledge_base: KnowledgeBase,
        query: Atom,
        *,
        strategy: str = "forward",
    ) -> bool:
        """Return whether the knowledge base entails a ground query."""
        if any(term.is_variable for term in query.arguments):
            raise ValueError("The engine requires a ground query")

        if strategy == "forward":
            derived_facts = forward_chain(
                knowledge_base,
                max_facts=self.max_facts,
            )
            return query in derived_facts

        if strategy == "backward":
            return backward_chain(
                knowledge_base,
                query,
                max_depth=self.max_depth,
            )

        raise ValueError(
            f"Unknown inference strategy: {strategy!r}. "
            "Choose 'forward' or 'backward'."
        )
