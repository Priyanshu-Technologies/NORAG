from typing import Protocol


class TokenCounter(Protocol):
    """Interface for estimating or counting prompt tokens."""

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...
