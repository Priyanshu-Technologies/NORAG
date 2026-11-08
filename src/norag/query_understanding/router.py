from typing import Any
from uuid import UUID

from norag.hybrid_retrieval.service import HybridRetriever
from norag.query_understanding.classifier import QueryClassifier
from norag.query_understanding.models import (
    QueryAnalysis,
    RoutedSearchResponse,
)


class QueryRouter:
    """Classify queries and select retrieval strategies."""

    def __init__(
        self,
        retriever: HybridRetriever,
        classifier: QueryClassifier | None = None,
    ) -> None:
        self.retriever = retriever
        self.classifier = classifier or QueryClassifier()

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query without executing retrieval."""

        return self.classifier.classify(query)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: UUID | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> RoutedSearchResponse:
        """Analyze a query and execute its selected retrieval strategy."""

        analysis = self.analyze(query)

        results = self.retriever.search(
            query=query,
            top_k=top_k,
            document_id=document_id,
            metadata_filters=metadata_filters,
            dense_weight=analysis.dense_weight,
            sparse_weight=analysis.sparse_weight,
        )

        return RoutedSearchResponse(
            analysis=analysis,
            results=results,
        )
