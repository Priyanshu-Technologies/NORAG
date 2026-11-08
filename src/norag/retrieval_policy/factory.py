from norag.hybrid_retrieval.service import HybridRetriever
from norag.query_understanding.classifier import QueryClassifier
from norag.reranking.service import RerankingService
from norag.retrieval_policy.policy import AdaptiveRetrievalPolicy
from norag.retrieval_policy.service import AdaptiveRetrievalService


def create_adaptive_retrieval_service(
    classifier: QueryClassifier,
    hybrid_retriever: HybridRetriever,
    reranking_service: RerankingService,
    policy: AdaptiveRetrievalPolicy | None = None,
) -> AdaptiveRetrievalService:
    """Connect NORAG's actual retrieval components."""

    def retrieve(query, plan):
        return hybrid_retriever.search(
            query=query,
            top_k=plan.candidate_k,
            dense_weight=plan.dense_weight,
            sparse_weight=plan.sparse_weight,
        )

    return AdaptiveRetrievalService(
        analyze=classifier.classify,
        retrieve=retrieve,
        rerank=reranking_service.rerank,
        policy=policy,
    )
