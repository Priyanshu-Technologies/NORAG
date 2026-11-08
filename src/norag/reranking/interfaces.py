from abc import ABC, abstractmethod
from collections.abc import Sequence


class RerankerProvider(ABC):
    """Contract for interchangeable neural reranking providers."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]: ...
