from collections.abc import Sequence

from fastembed import TextEmbedding

from norag.embeddings.interfaces import EmbeddingProvider


class FastEmbedProvider(EmbeddingProvider):
    """CPU-friendly embedding provider powered by FastEmbed."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        threads: int = 2,
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name

        self._model = TextEmbedding(
            model_name=model_name,
            threads=threads,
            cache_dir=cache_dir,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimensions(self) -> int:
        return self._model.embedding_size

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        vectors = self._model.embed(list(texts))

        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(self._model.query_embed(text))

        return [float(value) for value in vector]
