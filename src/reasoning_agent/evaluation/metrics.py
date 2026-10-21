from collections.abc import Sequence


def accuracy(
    predictions: Sequence[bool],
    labels: Sequence[bool],
) -> float:
    """Return accuracy as a fraction between 0 and 1."""
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have equal length")

    if not labels:
        raise ValueError("Cannot calculate accuracy on an empty set")

    correct = sum(
        prediction == label
        for prediction, label in zip(predictions, labels, strict=True)
    )

    return correct / len(labels)


def mean_inference_time(times: Sequence[float]) -> float:
    """Return average inference time in seconds."""
    if not times:
        raise ValueError("Cannot average an empty sequence")

    if any(time < 0 for time in times):
        raise ValueError("Inference times cannot be negative")

    return sum(times) / len(times)
