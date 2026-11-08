from typing import Literal

from norag.generation.service import GenerationService
from norag.verification.claim_extractor import ClaimExtractor
from norag.verification.providers.heuristic import (
    LexicalEvidenceVerifier,
)
from norag.verification.providers.ollama_judge import (
    OllamaEvidenceVerifier,
)
from norag.verification.service import ClaimVerificationService

VerificationStrategy = Literal["lexical", "ollama"]


def create_claim_verification_service(
    strategy: VerificationStrategy = "lexical",
    generation_service: GenerationService | None = None,
    minimum_overlap: float = 0.6,
    max_output_tokens: int = 200,
) -> ClaimVerificationService:
    if strategy == "lexical":
        verifier = LexicalEvidenceVerifier(
            minimum_overlap=minimum_overlap,
        )

    elif strategy == "ollama":
        if generation_service is None:
            raise ValueError("Ollama verification requires a generation service.")

        verifier = OllamaEvidenceVerifier(
            generation_service=generation_service,
            max_output_tokens=max_output_tokens,
        )

    else:
        raise ValueError(f"Unsupported verification strategy: {strategy}")

    return ClaimVerificationService(
        verifier=verifier,
        claim_extractor=ClaimExtractor(),
    )
