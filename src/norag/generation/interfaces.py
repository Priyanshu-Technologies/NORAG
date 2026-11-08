from abc import ABC, abstractmethod

from norag.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class GenerationProvider(ABC):
    """Interface implemented by every NORAG LLM provider."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate a response from the supplied request."""
