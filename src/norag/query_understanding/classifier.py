import re
from typing import ClassVar

from norag.query_understanding.models import (
    QueryAnalysis,
    QueryIntent,
    RetrievalRoute,
)


class QueryClassifier:
    """Classify queries using transparent heuristic rules."""

    COMPARATIVE_PATTERNS: ClassVar[list[str]] = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bdifference(?:s)?\b",
        r"\bdifferentiate\b",
        r"\bdistinguish\b",
        r"\bbetter than\b",
        r"\bpros and cons\b",
    ]

    MULTI_HOP_PATTERNS: ClassVar[list[str]] = [
        r"\band then\b",
        r"\bbased on\b",
        r"\baccording to\b",
        r"\bfrom that\b",
        r"\busing that\b",
        r"\bwhich .{1,120}\band\b",
        r"\bwhat .{1,120}\band\b",
        r"\bfind .{1,120}\bthen\b",
    ]

    EXACT_PATTERNS: ClassVar[list[str]] = [
        r"""["'`][^"'`]+["'`]""",
        r"\b[A-Z]{2,}(?:[-_][A-Z0-9]+)*\b",
        r"\b[A-Za-z]+\d+[A-Za-z0-9_-]*\b",
        r"\b[A-Za-z]+_[A-Za-z0-9_]+\b",
        r"\bv?\d+(?:\.\d+){1,3}\b",
        r"\b(?:HTTP|ERROR|ERR)[-_ ]?\d{3,}\b",
    ]

    @staticmethod
    def _matches(
        query: str,
        patterns: list[str],
    ) -> bool:
        return any(
            re.search(pattern, query, flags=re.IGNORECASE) for pattern in patterns
        )

    @staticmethod
    def _matches_exact(query: str) -> bool:
        return any(
            re.search(pattern, query) for pattern in QueryClassifier.EXACT_PATTERNS
        )

    def classify(self, query: str) -> QueryAnalysis:
        """Classify a query and determine retrieval weights."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        # Comparative intent takes priority over broad multi-hop patterns.
        if self._matches(query, self.COMPARATIVE_PATTERNS):
            return QueryAnalysis(
                query=query,
                intent=QueryIntent.COMPARATIVE,
                route=RetrievalRoute.HYBRID,
                dense_weight=1.0,
                sparse_weight=1.0,
                rule="comparative_pattern",
            )

        if self._matches(query, self.MULTI_HOP_PATTERNS):
            return QueryAnalysis(
                query=query,
                intent=QueryIntent.MULTI_HOP,
                route=RetrievalRoute.HYBRID,
                dense_weight=1.0,
                sparse_weight=1.0,
                rule="multi_hop_pattern",
            )

        # Definition questions should not be mistaken for identifier lookups.
        definition_patterns = [
            r"^\s*what\s+is\b",
            r"^\s*what\s+are\b",
            r"^\s*define\b",
            r"^\s*explain\b",
            r"^\s*describe\b",
        ]

        if self._matches(query, definition_patterns):
            return QueryAnalysis(
                query=query,
                intent=QueryIntent.SEMANTIC,
                route=RetrievalRoute.DENSE,
                dense_weight=1.0,
                sparse_weight=0.0,
                rule="definition_query",
            )

        if self._matches_exact(query):
            return QueryAnalysis(
                query=query,
                intent=QueryIntent.EXACT,
                route=RetrievalRoute.SPARSE,
                dense_weight=0.0,
                sparse_weight=1.0,
                rule="exact_identifier_pattern",
            )

        return QueryAnalysis(
            query=query,
            intent=QueryIntent.SEMANTIC,
            route=RetrievalRoute.DENSE,
            dense_weight=1.0,
            sparse_weight=0.0,
            rule="semantic_fallback",
        )
