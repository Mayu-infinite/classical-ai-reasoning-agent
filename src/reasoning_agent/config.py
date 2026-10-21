from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    max_facts: int = 100_000


DEFAULT_CONFIG = ReasoningConfig()
