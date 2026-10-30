from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class QueryExample:
    """One knowledge-base/query pair."""

    example_id: str
    context: tuple[str, ...]
    query: str
    label: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Prediction:
    """Prediction and optional diagnostic information."""

    example_id: str
    label: bool
    inference_time_seconds: float
    trace: tuple[str, ...] = ()
