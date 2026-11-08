import pytest

from norag.generation.models import GenerationResponse
from norag.verification.models import (
    EvidencePassage,
    EvidenceVerdict,
    ExtractedClaim,
)
from norag.verification.providers.ollama_judge import (
    OllamaEvidenceVerifier,
)


class FakeGenerationService:
    def __init__(self, content):
        self.content = content
        self.received_request = None

    def generate(self, request):
        self.received_request = request

        return GenerationResponse(
            content=self.content,
            provider_name="fake",
            model_name="fake-model",
        )


def create_claim():
    return ExtractedClaim(
        sentence_index=1,
        text="Dense retrieval captures semantic similarity.",
        cited_sources=[1],
    )


def create_evidence():
    return [
        EvidencePassage(
            source_number=1,
            chunk_id="chunk-1",
            text="Dense retrieval uses vectors to represent semantic meaning.",
        )
    ]


def test_supported_claim():
    generation = FakeGenerationService(
        '{"verdict":"supported",'
        '"supporting_sources":[1],'
        '"contradicting_sources":[],'
        '"explanation":"The evidence directly supports the claim."}'
    )

    verifier = OllamaEvidenceVerifier(generation)
    result = verifier.verify(create_claim(), create_evidence())

    assert result.verdict == EvidenceVerdict.SUPPORTED
    assert result.supporting_sources == [1]


def test_contradicted_claim():
    generation = FakeGenerationService(
        '{"verdict":"contradicted",'
        '"supporting_sources":[],'
        '"contradicting_sources":[1],'
        '"explanation":"The evidence explicitly conflicts with the claim."}'
    )

    verifier = OllamaEvidenceVerifier(generation)
    result = verifier.verify(create_claim(), create_evidence())

    assert result.verdict == EvidenceVerdict.CONTRADICTED
    assert result.contradicting_sources == [1]


def test_insufficient_evidence():
    generation = FakeGenerationService(
        '{"verdict":"insufficient_evidence",'
        '"supporting_sources":[],'
        '"contradicting_sources":[],'
        '"explanation":"The evidence does not establish the claim."}'
    )

    verifier = OllamaEvidenceVerifier(generation)
    result = verifier.verify(create_claim(), create_evidence())

    assert result.verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE


def test_missing_evidence_skips_generation():
    generation = FakeGenerationService("{}")

    verifier = OllamaEvidenceVerifier(generation)
    result = verifier.verify(create_claim(), [])

    assert result.verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE
    assert generation.received_request is None


def test_invalid_json_is_rejected():
    generation = FakeGenerationService("This is not JSON.")

    verifier = OllamaEvidenceVerifier(generation)

    with pytest.raises(ValueError, match="invalid verification JSON"):
        verifier.verify(create_claim(), create_evidence())


def test_invalid_verdict_is_rejected():
    generation = FakeGenerationService('{"verdict":"maybe","explanation":"Uncertain."}')

    verifier = OllamaEvidenceVerifier(generation)

    with pytest.raises(ValueError, match="invalid verification verdict"):
        verifier.verify(create_claim(), create_evidence())


def test_generation_request_uses_zero_temperature():
    generation = FakeGenerationService(
        '{"verdict":"supported",'
        '"supporting_sources":[1],'
        '"contradicting_sources":[],'
        '"explanation":"Evidence matches."}'
    )

    verifier = OllamaEvidenceVerifier(generation)
    verifier.verify(create_claim(), create_evidence())

    assert generation.received_request.temperature == 0.0


def test_invalid_output_limit_is_rejected():
    with pytest.raises(ValueError):
        OllamaEvidenceVerifier(
            FakeGenerationService("{}"),
            max_output_tokens=0,
        )


def test_rejects_invented_source_reference():
    generation = FakeGenerationService(
        '{"verdict":"supported",'
        '"supporting_sources":[99],'
        '"contradicting_sources":[],'
        '"explanation":"Evidence supports the claim."}'
    )

    verifier = OllamaEvidenceVerifier(generation)

    with pytest.raises(ValueError, match="not supplied"):
        verifier.verify(create_claim(), create_evidence())


def test_rejects_supported_verdict_without_supporting_source():
    generation = FakeGenerationService(
        '{"verdict":"supported",'
        '"supporting_sources":[],'
        '"contradicting_sources":[],'
        '"explanation":"Evidence supports the claim."}'
    )

    verifier = OllamaEvidenceVerifier(generation)

    with pytest.raises(ValueError, match="requires supporting sources"):
        verifier.verify(create_claim(), create_evidence())
