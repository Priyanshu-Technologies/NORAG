import math

from norag.chunking.models import Chunk
from norag.embeddings.interfaces import EmbeddingProvider
from norag.embeddings.models import EmbeddedChunk


class EmbeddingService:
    """Generate and validate embeddings for NORAG chunks."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.provider = provider
        self.batch_size = batch_size

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.provider.embedding_dimensions:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.provider.embedding_dimensions}, "
                f"received {len(vector)}"
            )

        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Provider returned an invalid embedding vector")

        return vector

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed chunks in batches while preserving their original order."""

        results: list[EmbeddedChunk] = []

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]

            for chunk in batch:
                if not chunk.text.strip():
                    raise ValueError(f"Chunk {chunk.id} contains empty text")

            vectors = self.provider.embed_documents([chunk.text for chunk in batch])

            if len(vectors) != len(batch):
                raise RuntimeError(
                    "Embedding provider returned an unexpected number of vectors"
                )

            for chunk, vector in zip(batch, vectors, strict=True):
                vector = self._validate_vector(vector)

                results.append(
                    EmbeddedChunk(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        index=chunk.index,
                        text=chunk.text,
                        vector=vector,
                        dimensions=len(vector),
                        model_name=self.provider.model_name,
                        source_element_ids=chunk.source_element_ids,
                        source_locations=chunk.source_locations,
                        metadata=chunk.metadata,
                    )
                )

        return results

    def embed_query(self, query: str) -> list[float]:
        """Generate an embedding for a search query."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        vector = self.provider.embed_query(query)

        return self._validate_vector(vector)
