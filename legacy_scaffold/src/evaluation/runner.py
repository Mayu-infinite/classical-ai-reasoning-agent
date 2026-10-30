from time import perf_counter

from reasoning_agent.inference.engine import ReasoningEngine
from reasoning_agent.logic.ast import Atom
from reasoning_agent.logic.knowledge_base import KnowledgeBase


def run_query(
    engine: ReasoningEngine,
    knowledge_base: KnowledgeBase,
    query: Atom,
) -> tuple[bool, float]:
    """Run one query and return (prediction, elapsed_seconds)."""
    start = perf_counter()
    prediction = engine.entails(knowledge_base, query)
    elapsed = perf_counter() - start

    return prediction, elapsed
