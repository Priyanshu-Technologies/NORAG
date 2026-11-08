from norag.reranking.providers.fastembed import FastEmbedReranker
from norag.reranking.service import RerankingService


def create_reranking_service(
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
) -> RerankingService:
    """Create NORAG's default neural reranking service."""

    provider = FastEmbedReranker(
        model_name=model_name,
    )

    return RerankingService(
        provider=provider,
    )
