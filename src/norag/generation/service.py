from norag.generation.interfaces import GenerationProvider
from norag.generation.models import (
    GenerationRequest,
    GenerationResponse,
)


class GenerationService:
    """Coordinate generation through a configured LLM provider."""

    def __init__(self, provider: GenerationProvider) -> None:
        self.provider = provider

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        response = self.provider.generate(request)

        if not isinstance(response, GenerationResponse):
            raise TypeError("Generation provider must return GenerationResponse")

        if not response.content.strip():
            raise RuntimeError("Generation provider returned empty content")

        return response
