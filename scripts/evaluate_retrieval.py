from uuid import uuid4

from norag.chunking.models import Chunk
from norag.embeddings.factory import create_embedding_service
from norag.evaluation import EvaluationCase, RetrievalEvaluator
from norag.hybrid_retrieval.service import HybridRetriever
from norag.reranking import create_reranking_service
from norag.sparse_retrieval import BM25Index
from norag.vector_store import InMemoryVectorIndex

documents = [
    "YOLO11 supports object detection and instance segmentation.",
    "BERT is a transformer-based language representation model.",
    "Instance segmentation identifies individual objects and generates a mask for each object.",
    "Python is a general-purpose programming language.",
    "YOLO11 is an object detection model used in computer vision.",
    "Retrieval augmented generation combines information retrieval with language models.",
]

chunks = [
    Chunk(
        document_id=uuid4(),
        index=i,
        text=text,
        token_count=len(text.split()),
    )
    for i, text in enumerate(documents)
]

embedding_service = create_embedding_service()
embedded_chunks = embedding_service.embed_chunks(chunks)

dense_index = InMemoryVectorIndex(dimensions=384)
sparse_index = BM25Index()

dense_index.add_many(embedded_chunks)
sparse_index.add_many(chunks)

hybrid = HybridRetriever(
    embedding_service=embedding_service,
    dense_index=dense_index,
    sparse_index=sparse_index,
)

reranker = create_reranking_service()
evaluator = RetrievalEvaluator()

cases = [
    EvaluationCase(
        query="How does YOLO11 perform instance segmentation?",
        relevant_chunk_ids={
            chunks[0].id,
            chunks[2].id,
            chunks[4].id,
        },
    ),
    EvaluationCase(
        query="What is BERT used for?",
        relevant_chunk_ids={chunks[1].id},
    ),
    EvaluationCase(
        query="How does RAG combine retrieval and language models?",
        relevant_chunk_ids={chunks[5].id},
    ),
]


def dense_search(query, k):
    vector = embedding_service.embed_query(query)
    results = dense_index.search(vector, top_k=k)
    return [result.chunk.chunk_id for result in results]


def sparse_search(query, k):
    results = sparse_index.search(query, top_k=k)
    return [result.chunk.id for result in results]


def hybrid_search(query, k):
    results = hybrid.search(query, top_k=k)
    return [result.chunk.id for result in results]


def reranked_search(query, k):
    candidates = hybrid.search(
        query,
        top_k=k,
        candidate_k=len(chunks),
    )

    results = reranker.rerank(
        query,
        candidates,
        top_k=k,
    )

    return [result.result.chunk.id for result in results]


systems = {
    "Dense": dense_search,
    "Sparse": sparse_search,
    "Hybrid": hybrid_search,
    "Hybrid + Reranking": reranked_search,
}

print("\nNORAG RETRIEVAL EVALUATION")
print("=" * 75)

for name, search_function in systems.items():
    report = evaluator.evaluate(
        cases=cases,
        search_fn=search_function,
        k=3,
    )

    metrics = report.average_metrics

    print(f"\n{name}")
    print(f"Recall@3:    {metrics.recall_at_k:.4f}")
    print(f"Precision@3: {metrics.precision_at_k:.4f}")
    print(f"MRR@3:       {metrics.mrr_at_k:.4f}")
    print(f"NDCG@3:      {metrics.ndcg_at_k:.4f}")
