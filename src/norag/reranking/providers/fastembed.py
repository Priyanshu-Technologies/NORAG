from collections.abc import Sequence

from fastembed.rerank.cross_encoder import TextCrossEncoder

from norag.reranking.interfaces import RerankerProvider


class FastEmbedReranker(RerankerProvider):
    """Local cross-encoder reranker powered by FastEmbed."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self._model_name = model_name

        self._model = TextCrossEncoder(
            model_name=model_name,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """Score documents against a query in their original order."""

        if not documents:
            return []

        scores = self._model.rerank(
            query,
            list(documents),
        )

        return [float(score) for score in scores]
