import pytest

from norag.verification.claim_extractor import ClaimExtractor
from norag.verification.models import (
    EvidencePassage,
    EvidenceVerdict,
)
from norag.verification.providers.heuristic import (
    LexicalEvidenceVerifier,
)


def test_extracts_claims_and_citations():
    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Dense retrieval captures semantic similarity [Source 1]. "
        "BM25 supports lexical matching [Source 2]."
    )

    assert len(claims) == 2
    assert claims[0].cited_sources == [1]
    assert claims[1].cited_sources == [2]
    assert "[Source" not in claims[0].text


def test_extracts_multiple_citations():
    extractor = ClaimExtractor()

    claims = extractor.extract(
        "Hybrid retrieval combines both methods [Source 1] [Source 2]."
    )

    assert claims[0].cited_sources == [1, 2]


def test_extracts_uncited_claim():
    extractor = ClaimExtractor()

    claims = extractor.extract("Dense retrieval captures semantic similarity.")

    assert len(claims) == 1
    assert claims[0].cited_sources == []


def test_empty_answer_returns_no_claims():
    extractor = ClaimExtractor()

    assert extractor.extract("  ") == []


def test_lexical_verifier_supports_matching_evidence():
    from norag.verification.models import ExtractedClaim

    verifier = LexicalEvidenceVerifier()

    claim = ExtractedClaim(
        sentence_index=1,
        text="Dense retrieval captures semantic similarity",
        cited_sources=[1],
    )

    evidence = [
        EvidencePassage(
            source_number=1,
            chunk_id="chunk-1",
            text=(
                "Dense retrieval captures semantic similarity "
                "using numerical vector representations."
            ),
        )
    ]

    result = verifier.verify(claim, evidence)

    assert result.verdict == EvidenceVerdict.SUPPORTED
    assert result.supporting_sources == [1]
    assert result.method == "lexical_overlap_heuristic"


def test_lexical_verifier_rejects_weak_overlap():
    from norag.verification.models import ExtractedClaim

    verifier = LexicalEvidenceVerifier()

    claim = ExtractedClaim(
        sentence_index=1,
        text="BM25 is a neural image classification architecture",
        cited_sources=[1],
    )

    evidence = [
        EvidencePassage(
            source_number=1,
            chunk_id="chunk-1",
            text="BM25 is a sparse lexical retrieval algorithm.",
        )
    ]

    result = verifier.verify(claim, evidence)

    assert result.verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE
    assert result.contradicting_sources == []


def test_uncited_claim_has_insufficient_evidence():
    from norag.verification.models import ExtractedClaim

    verifier = LexicalEvidenceVerifier()

    claim = ExtractedClaim(
        sentence_index=1,
        text="Dense retrieval captures semantic similarity",
    )

    result = verifier.verify(claim, [])

    assert result.verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE


def test_invalid_overlap_threshold_is_rejected():
    with pytest.raises(ValueError):
        LexicalEvidenceVerifier(minimum_overlap=0.0)


def test_report_calculates_support_rate():
    from norag.verification.models import (
        ClaimEvidenceResult,
        ClaimVerificationReport,
        ExtractedClaim,
    )

    claim = ExtractedClaim(
        sentence_index=1,
        text="Dense retrieval captures semantic similarity",
        cited_sources=[1],
    )

    result = ClaimEvidenceResult(
        claim=claim,
        verdict=EvidenceVerdict.SUPPORTED,
        supporting_sources=[1],
        explanation="Lexical match.",
        method="lexical_overlap_heuristic",
    )

    report = ClaimVerificationReport(
        total_claims=2,
        supported_claims=1,
        contradicted_claims=0,
        insufficient_claims=1,
        results=[result],
    )

    assert report.support_rate == 0.5
    assert report.has_unsupported_claims
