from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TraceStep:
    rule_description: str
    premises: tuple[str, ...]
    conclusion: str


@dataclass(slots=True)
class ReasoningTrace:
    steps: list[TraceStep]

    def add_step(
        self,
        rule_description: str,
        premises: tuple[str, ...],
        conclusion: str,
    ) -> None:
        self.steps.append(
            TraceStep(
                rule_description=rule_description,
                premises=premises,
                conclusion=conclusion,
            )
        )
