from enum import StrEnum

from pydantic import BaseModel, Field

# Citation validation models


class CitationStatus(StrEnum):
    VALID = "valid"
    UNCITED = "uncited"
    INVALID = "invalid"


class CitationIssue(BaseModel):
    sentence_index: int = Field(ge=1)
    citation: str
    reason: str


class VerifiedSentence(BaseModel):
    sentence_index: int = Field(ge=1)
    text: str
    cited_sources: list[int] = Field(default_factory=list)
    status: CitationStatus


class CitationValidationReport(BaseModel):
    total_sentences: int = Field(ge=0)
    cited_sentences: int = Field(ge=0)
    uncited_sentences: int = Field(ge=0)
    invalid_citation_count: int = Field(ge=0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    sentences: list[VerifiedSentence] = Field(default_factory=list)
    issues: list[CitationIssue] = Field(default_factory=list)

    @property
    def citations_are_valid(self) -> bool:
        return self.invalid_citation_count == 0

    @property
    def coverage_is_complete(self) -> bool:
        return self.uncited_sentences == 0


# Claim verification models


class EvidenceVerdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidencePassage(BaseModel):
    source_number: int = Field(ge=1)
    chunk_id: str
    text: str = Field(min_length=1)
    section: str | None = None


class ExtractedClaim(BaseModel):
    sentence_index: int = Field(ge=1)
    text: str = Field(min_length=1)
    cited_sources: list[int] = Field(default_factory=list)


class ClaimEvidenceResult(BaseModel):
    claim: ExtractedClaim
    verdict: EvidenceVerdict
    supporting_sources: list[int] = Field(default_factory=list)
    contradicting_sources: list[int] = Field(default_factory=list)
    explanation: str
    method: str


class ClaimVerificationReport(BaseModel):
    total_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    contradicted_claims: int = Field(ge=0)
    insufficient_claims: int = Field(ge=0)
    results: list[ClaimEvidenceResult] = Field(default_factory=list)

    @property
    def support_rate(self) -> float:
        if self.total_claims == 0:
            return 0.0

        return self.supported_claims / self.total_claims

    @property
    def has_unsupported_claims(self) -> bool:
        return self.contradicted_claims > 0 or self.insufficient_claims > 0
