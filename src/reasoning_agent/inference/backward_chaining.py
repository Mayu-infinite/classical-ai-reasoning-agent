"""Backward-chaining inference.

Implementation is intentionally deferred until the supported logical
fragment and query semantics have been validated against the datasets.

The first working inference baseline is forward chaining over positive
Horn rules.
"""


class BackwardChainingNotImplemented(NotImplementedError):
    """Raised when backward chaining is requested before implementation."""
