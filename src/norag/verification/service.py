from collections.abc import Sequence

from norag.context.models import AssembledContext
from norag.prompting.models import PromptSource
from norag.verification.claim_extractor import ClaimExtractor
from norag.verification.interfaces import EvidenceVerifier
from norag.verification.models import (
    ClaimVerificationReport,
    EvidencePassage,
    EvidenceVerdict,
)


class ClaimVerificationService:
    def __init__(
        self,
        verifier: EvidenceVerifier,
        claim_extractor: ClaimExtractor | None = None,
    ) -> None:
        self.verifier = verifier
        self.claim_extractor = claim_extractor or ClaimExtractor()

    def verify(
        self,
        answer: str,
        sources: Sequence[PromptSource],
        context: AssembledContext,
    ) -> ClaimVerificationReport:
        claims = self.claim_extractor.extract(answer)

        context_by_position = {item.position: item for item in context.items}

        evidence_by_source: dict[int, EvidencePassage] = {}

        for source in sources:
            context_item = context_by_position.get(source.position)

            if context_item is None:
                continue

            chunk = context_item.result.result.chunk

            evidence_by_source[source.position] = EvidencePassage(
                source_number=source.position,
                chunk_id=source.chunk_id,
                text=chunk.text,
                section=source.section,
            )

        results = []

        for claim in claims:
            cited_evidence = [
                evidence_by_source[source_number]
                for source_number in claim.cited_sources
                if source_number in evidence_by_source
            ]

            results.append(
                self.verifier.verify(
                    claim=claim,
                    evidence=cited_evidence,
                )
            )

        supported = sum(
            result.verdict == EvidenceVerdict.SUPPORTED for result in results
        )
        contradicted = sum(
            result.verdict == EvidenceVerdict.CONTRADICTED for result in results
        )
        insufficient = sum(
            result.verdict == EvidenceVerdict.INSUFFICIENT_EVIDENCE
            for result in results
        )

        return ClaimVerificationReport(
            total_claims=len(results),
            supported_claims=supported,
            contradicted_claims=contradicted,
            insufficient_claims=insufficient,
            results=results,
        )
