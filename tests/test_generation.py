import pytest
from pydantic import ValidationError

from norag.generation import (
    GenerationProvider,
    GenerationRequest,
    GenerationResponse,
    GenerationService,
)


class FakeGenerationProvider(GenerationProvider):
    """Deterministic provider for testing the generation contract."""

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return GenerationResponse(
            content=f"Generated answer for: {request.user_prompt}",
            provider_name=self.provider_name,
            model_name=self.model_name,
            input_tokens=20,
            output_tokens=15,
            finish_reason="stop",
        )


def test_generation_request_uses_defaults() -> None:
    request = GenerationRequest(user_prompt="Explain RAG")

    assert request.temperature == 0.2
    assert request.max_output_tokens == 512
    assert request.system_prompt == ""


def test_generation_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(user_prompt="   ")


def test_generation_request_rejects_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            user_prompt="Explain RAG",
            temperature=3.0,
        )


def test_generation_request_rejects_invalid_max_tokens() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            user_prompt="Explain RAG",
            max_output_tokens=0,
        )


def test_generation_response_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        GenerationResponse(
            content="   ",
            provider_name="fake",
            model_name="fake-model",
        )


def test_generation_service_returns_provider_response() -> None:
    service = GenerationService(FakeGenerationProvider())

    request = GenerationRequest(user_prompt="Explain hybrid retrieval")
    response = service.generate(request)

    assert isinstance(response, GenerationResponse)
    assert "hybrid retrieval" in response.content
    assert response.provider_name == "fake"
    assert response.model_name == "fake-model"


def test_generation_service_preserves_usage_information() -> None:
    service = GenerationService(FakeGenerationProvider())

    response = service.generate(GenerationRequest(user_prompt="Explain BM25"))

    assert response.input_tokens == 20
    assert response.output_tokens == 15
    assert response.finish_reason == "stop"


def test_generation_service_rejects_invalid_provider_response() -> None:
    class InvalidProvider(FakeGenerationProvider):
        def generate(self, request: GenerationRequest):
            return "not a GenerationResponse"

    service = GenerationService(InvalidProvider())

    with pytest.raises(TypeError, match="GenerationResponse"):
        service.generate(GenerationRequest(user_prompt="Test"))
