from norag.answering.models import AnswerRequest, AnswerResponse
from norag.context.assembler import ContextAssembler
from norag.generation.service import GenerationService
from norag.prompting.builder import PromptBuilder
from norag.retrieval_policy.service import AdaptiveRetrievalService
from norag.verification.claim_extractor import ClaimExtractor
from norag.verification.providers.heuristic import (
    LexicalEvidenceVerifier,
)
from norag.verification.service import ClaimVerificationService
from norag.verification.validator import CitationValidator


class AnsweringService:
    def __init__(
        self,
        retrieval_service: AdaptiveRetrievalService,
        context_assembler: ContextAssembler,
        prompt_builder: PromptBuilder,
        generation_service: GenerationService,
        citation_validator: CitationValidator | None = None,
        claim_verification_service: ClaimVerificationService | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.context_assembler = context_assembler
        self.prompt_builder = prompt_builder
        self.generation_service = generation_service

        self.citation_validator = citation_validator or CitationValidator()

        self.claim_verification_service = (
            claim_verification_service
            or ClaimVerificationService(
                verifier=LexicalEvidenceVerifier(),
                claim_extractor=ClaimExtractor(),
            )
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
    ) -> AnswerResponse:
        request = AnswerRequest(
            query=query,
            top_k=top_k,
        )

        retrieval = self.retrieval_service.search(
            query=request.query,
            top_k=request.top_k,
        )

        context = self.context_assembler.assemble(retrieval.results)

        prompt = self.prompt_builder.build(
            query=request.query,
            context=context,
        )

        generation = self.generation_service.generate(prompt.request)

        citation_validation = self.citation_validator.validate(
            answer=generation.content,
            sources=prompt.sources,
        )

        claim_verification = self.claim_verification_service.verify(
            answer=generation.content,
            sources=prompt.sources,
            context=context,
        )

        return AnswerResponse(
            query=request.query,
            answer=generation.content,
            retrieval=retrieval,
            context=context,
            prompt=prompt,
            generation=generation,
            citation_validation=citation_validation,
            claim_verification=claim_verification,
        )
