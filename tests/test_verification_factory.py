import pytest

from norag.verification.factory import (
    create_claim_verification_service,
)
from norag.verification.providers.heuristic import (
    LexicalEvidenceVerifier,
)
from norag.verification.providers.ollama_judge import (
    OllamaEvidenceVerifier,
)


class FakeGenerationService:
    pass


def test_factory_defaults_to_lexical_verification():
    service = create_claim_verification_service()

    assert isinstance(service.verifier, LexicalEvidenceVerifier)


def test_factory_creates_ollama_verifier():
    service = create_claim_verification_service(
        strategy="ollama",
        generation_service=FakeGenerationService(),
    )

    assert isinstance(service.verifier, OllamaEvidenceVerifier)


def test_ollama_strategy_requires_generation_service():
    with pytest.raises(ValueError, match="requires a generation service"):
        create_claim_verification_service(strategy="ollama")


def test_factory_rejects_unsupported_strategy():
    with pytest.raises(ValueError, match="Unsupported verification strategy"):
        create_claim_verification_service(strategy="unsupported")
