
from reasoning_agent.config import DEFAULT_CONFIG
from reasoning_agent.inference.engine import ReasoningEngine
from reasoning_agent.logic.ast import Atom
from reasoning_agent.logic.knowledge_base import KnowledgeBase


def predict_from_logic(
    knowledge_base: KnowledgeBase,
    query: Atom,
    *,
    strategy: str = "forward",
) -> bool:
    """Predict for an already-parsed positive Horn knowledge base."""
    engine = ReasoningEngine(
        max_facts=DEFAULT_CONFIG.max_facts,
    )
    return engine.entails(
        knowledge_base,
        query,
        strategy=strategy,
    )
