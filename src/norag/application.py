from dataclasses import dataclass

from norag.answering.service import AnsweringService
from norag.chunking.models import ChunkingConfig
from norag.chunking.strategies.structural import StructuralChunker
from norag.config import Settings, get_settings
from norag.context.assembler import ContextAssembler
from norag.context.models import ContextConfig
from norag.embeddings.factory import create_embedding_service
from norag.embeddings.service import EmbeddingService
from norag.generation.providers.ollama import OllamaProvider
from norag.generation.service import GenerationService
from norag.hybrid_retrieval.service import HybridRetriever
from norag.ingestion.factory import create_ingestion_service
from norag.ingestion.service import IngestionService
from norag.prompting.builder import PromptBuilder
from norag.prompting.models import PromptConfig
from norag.query_understanding.classifier import QueryClassifier
from norag.reranking.factory import create_reranking_service
from norag.reranking.service import RerankingService
from norag.retrieval_policy.factory import (
    create_adaptive_retrieval_service,
)
from norag.retrieval_policy.service import AdaptiveRetrievalService
from norag.sparse_retrieval.bm25 import BM25Index
from norag.vector_store.memory import InMemoryVectorIndex
from norag.verification.factory import create_claim_verification_service
from norag.verification.service import ClaimVerificationService


@dataclass
class ApplicationServices:
    """Container holding all initialized NORAG application services."""

    ingestion_service: IngestionService
    chunker: StructuralChunker
    embedding_service: EmbeddingService

    dense_index: InMemoryVectorIndex
    sparse_index: BM25Index

    hybrid_retriever: HybridRetriever
    reranking_service: RerankingService
    retrieval_service: AdaptiveRetrievalService

    context_assembler: ContextAssembler
    prompt_builder: PromptBuilder

    generation_service: GenerationService
    claim_verification_service: ClaimVerificationService

    answering_service: AnsweringService


def create_application_services(
    settings: Settings | None = None,
) -> ApplicationServices:
    """Construct and connect the complete NORAG application."""

    settings = settings or get_settings()

    # Document ingestion and chunking
    ingestion_service = create_ingestion_service()

    chunker = StructuralChunker(
        ChunkingConfig(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    )

    # Embedding and indexing
    embedding_service = create_embedding_service()

    dense_index = InMemoryVectorIndex()
    sparse_index = BM25Index()

    # Hybrid retrieval
    hybrid_retriever = HybridRetriever(
        embedding_service=embedding_service,
        dense_index=dense_index,
        sparse_index=sparse_index,
    )

    # Query understanding and reranking
    classifier = QueryClassifier()
    reranking_service = create_reranking_service()

    retrieval_service = create_adaptive_retrieval_service(
        classifier=classifier,
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
    )

    # Context assembly and prompt construction
    context_assembler = ContextAssembler(
        ContextConfig(max_tokens=settings.context_max_tokens)
    )
    prompt_builder = PromptBuilder(
        PromptConfig(
            max_input_tokens=settings.prompt_max_input_tokens,
            temperature=settings.generation_temperature,
            max_output_tokens=settings.prompt_max_output_tokens,
        )
    )

    # LLM generation
    generation_provider = OllamaProvider(
        model_name=settings.ollama_model,
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout_seconds,
        think=settings.ollama_think,
    )

    generation_service = GenerationService(
        provider=generation_provider,
    )

    # Configurable claim verification
    claim_verification_service = create_claim_verification_service(
        strategy=settings.verification_strategy,
        generation_service=generation_service,
    )

    # Complete answering pipeline
    answering_service = AnsweringService(
        retrieval_service=retrieval_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        generation_service=generation_service,
        claim_verification_service=claim_verification_service,
    )

    return ApplicationServices(
        ingestion_service=ingestion_service,
        chunker=chunker,
        embedding_service=embedding_service,
        dense_index=dense_index,
        sparse_index=sparse_index,
        hybrid_retriever=hybrid_retriever,
        reranking_service=reranking_service,
        retrieval_service=retrieval_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        generation_service=generation_service,
        claim_verification_service=claim_verification_service,
        answering_service=answering_service,
    )
