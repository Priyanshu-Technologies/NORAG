from abc import ABC, abstractmethod
from collections.abc import Sequence

from norag.verification.models import (
    ClaimEvidenceResult,
    EvidencePassage,
    ExtractedClaim,
)


class EvidenceVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        claim: ExtractedClaim,
        evidence: Sequence[EvidencePassage],
    ) -> ClaimEvidenceResult:
        """Evaluate a claim against its supplied evidence."""
        raise NotImplementedError
