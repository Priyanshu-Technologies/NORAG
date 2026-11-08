from pydantic import BaseModel, Field, field_validator

from norag.context.models import AssembledContext
from norag.generation.models import GenerationResponse
from norag.prompting.models import PromptConstruction
from norag.retrieval_policy.service import AdaptiveRetrievalResponse
from norag.verification.models import CitationValidationReport, ClaimVerificationReport


class AnswerRequest(BaseModel):
    """Input for an end-to-end NORAG question."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be empty.")

        return value.strip()


class AnswerResponse(BaseModel):
    """Answer and trace of the complete generation pipeline."""

    query: str
    answer: str
    retrieval: AdaptiveRetrievalResponse
    context: AssembledContext
    prompt: PromptConstruction
    generation: GenerationResponse
    citation_validation: CitationValidationReport
    claim_verification: ClaimVerificationReport
