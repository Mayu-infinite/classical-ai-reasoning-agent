from collections.abc import Mapping
from typing import Any

from reasoning_agent.data.schemas import QueryExample


def generic_record_to_example(
    record: Mapping[str, Any],
    *,
    context_key: str,
    query_key: str,
    label_key: str | None = None,
    id_key: str | None = None,
) -> QueryExample:
    """Convert a record using explicitly supplied field names."""
    if context_key not in record:
        raise ValueError(f"Missing context field: {context_key}")

    if query_key not in record:
        raise ValueError(f"Missing query field: {query_key}")

    raw_context = record[context_key]
    raw_query = record[query_key]

    if isinstance(raw_context, str):
        context = (raw_context,)
    elif isinstance(raw_context, (list, tuple)) and all(
        isinstance(item, str) for item in raw_context
    ):
        context = tuple(raw_context)
    else:
        raise ValueError("Context must be a string or a list of strings")

    if not isinstance(raw_query, str) or not raw_query.strip():
        raise ValueError("Query must be a non-empty string")

    label = None
    if label_key is not None:
        raw_label = record.get(label_key)
        if raw_label is not None:
            if isinstance(raw_label, bool):
                label = raw_label
            elif isinstance(raw_label, str):
                normalized = raw_label.strip().lower()
                if normalized in {"true", "1"}:
                    label = True
                elif normalized in {"false", "0"}:
                    label = False
                else:
                    raise ValueError(f"Unrecognized label: {raw_label!r}")
            else:
                raise ValueError(f"Unrecognized label: {raw_label!r}")

    example_id = str(record.get(id_key, "")) if id_key else ""
    if not example_id:
        example_id = "unassigned"

    return QueryExample(
        example_id=example_id,
        context=context,
        query=raw_query.strip(),
        label=label,
        metadata=dict(record),
    )
