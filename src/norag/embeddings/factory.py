from norag.embeddings.providers.fastembed import FastEmbedProvider
from norag.embeddings.service import EmbeddingService


def create_embedding_service(
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 16,
) -> EmbeddingService:
    """Create NORAG's default embedding service."""

    provider = FastEmbedProvider(
        model_name=model_name,
        threads=2,
    )

    return EmbeddingService(
        provider=provider,
        batch_size=batch_size,
    )
