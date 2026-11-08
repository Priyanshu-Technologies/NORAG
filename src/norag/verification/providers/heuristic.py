import re
from collections.abc import Sequence
from typing import ClassVar

from norag.verification.interfaces import EvidenceVerifier
from norag.verification.models import (
    ClaimEvidenceResult,
    EvidencePassage,
    EvidenceVerdict,
    ExtractedClaim,
)


class LexicalEvidenceVerifier(EvidenceVerifier):
    STOPWORDS: ClassVar[set[str]] = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "this",
        "those",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
    }

    def __init__(self, minimum_overlap: float = 0.6) -> None:
        if not 0.0 < minimum_overlap <= 1.0:
            raise ValueError("minimum_overlap must be greater than 0 and at most 1.")

        self.minimum_overlap = minimum_overlap

    def _tokens(self, text: str) -> set[str]:
        tokens = set(re.findall(r"(?u)[^\W_]+", text.casefold()))

        return tokens - self.STOPWORDS

    def verify(
        self,
        claim: ExtractedClaim,
        evidence: Sequence[EvidencePassage],
    ) -> ClaimEvidenceResult:
        claim_tokens = self._tokens(claim.text)

        if not claim.cited_sources:
            return ClaimEvidenceResult(
                claim=claim,
                verdict=EvidenceVerdict.INSUFFICIENT_EVIDENCE,
                explanation="The claim has no source citations.",
                method="lexical_overlap_heuristic",
            )

        relevant_evidence = [
            passage
            for passage in evidence
            if passage.source_number in claim.cited_sources
        ]

        if not relevant_evidence or not claim_tokens:
            return ClaimEvidenceResult(
                claim=claim,
                verdict=EvidenceVerdict.INSUFFICIENT_EVIDENCE,
                explanation=(
                    "No usable cited evidence was available for lexical comparison."
                ),
                method="lexical_overlap_heuristic",
            )

        evidence_by_source: dict[int, set[str]] = {}

        for passage in relevant_evidence:
            evidence_by_source.setdefault(
                passage.source_number,
                set(),
            ).update(self._tokens(passage.text))

        source_scores: dict[int, float] = {}

        for source_number, evidence_tokens in evidence_by_source.items():
            overlap = claim_tokens & evidence_tokens
            source_scores[source_number] = len(overlap) / len(claim_tokens)

        best_source, best_score = max(
            source_scores.items(),
            key=lambda item: item[1],
        )

        if best_score >= self.minimum_overlap:
            verdict = EvidenceVerdict.SUPPORTED
            supporting_sources = [best_source]
            explanation = (
                "The cited passage has sufficient lexical overlap "
                f"with the claim ({best_score:.2f}). "
                "This is a heuristic match, not proof of entailment."
            )
        else:
            verdict = EvidenceVerdict.INSUFFICIENT_EVIDENCE
            supporting_sources = []
            explanation = (
                "The cited evidence has insufficient lexical overlap "
                f"with the claim ({best_score:.2f}). "
                "This does not establish that the claim is false."
            )

        return ClaimEvidenceResult(
            claim=claim,
            verdict=verdict,
            supporting_sources=supporting_sources,
            explanation=explanation,
            method="lexical_overlap_heuristic",
        )
