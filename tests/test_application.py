from norag.application import (
    ApplicationServices,
    create_application_services,
)
from norag.config import Settings


def test_application_factory_creates_all_services():
    settings = Settings(_env_file=None)

    services = create_application_services(settings)

    assert isinstance(services, ApplicationServices)

    assert services.ingestion_service is not None
    assert services.chunker is not None
    assert services.embedding_service is not None

    assert services.dense_index is not None
    assert services.sparse_index is not None

    assert services.hybrid_retriever is not None
    assert services.reranking_service is not None
    assert services.retrieval_service is not None

    assert services.context_assembler is not None
    assert services.prompt_builder is not None

    assert services.generation_service is not None
    assert services.claim_verification_service is not None
    assert services.answering_service is not None


def test_application_uses_configured_verification_strategy():
    settings = Settings(
        _env_file=None,
        NORAG_VERIFICATION_STRATEGY="lexical",
    )

    services = create_application_services(settings)

    assert (
        services.claim_verification_service.verifier.__class__.__name__
        == "LexicalEvidenceVerifier"
    )
