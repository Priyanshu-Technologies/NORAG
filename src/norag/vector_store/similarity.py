import math
from collections.abc import Sequence


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if not left or not right:
        raise ValueError("Vectors cannot be empty")

    if len(left) != len(right):
        raise ValueError("Vectors must have equal dimensions")

    if not all(math.isfinite(value) for value in left):
        raise ValueError("Left vector contains non-finite values")

    if not all(math.isfinite(value) for value in right):
        raise ValueError("Right vector contains non-finite values")

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        raise ValueError("Zero vectors cannot be compared")

    return dot_product / (left_norm * right_norm)
