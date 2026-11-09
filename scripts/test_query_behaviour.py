from pathlib import Path
from tempfile import TemporaryDirectory

from norag.chunking.strategies.structural import StructuralChunker
from norag.embeddings.factory import create_embedding_service
from norag.hybrid_retrieval.service import HybridRetriever
from norag.ingestion.factory import create_ingestion_service
from norag.query_understanding.classifier import QueryClassifier
from norag.query_understanding.models import QueryIntent
from norag.reranking.factory import create_reranking_service
from norag.retrieval_policy.factory import (
    create_adaptive_retrieval_service,
)
from norag.sparse_retrieval.bm25 import BM25Index
from norag.vector_store.memory import InMemoryVectorIndex

SAMPLE_DOCUMENT = """
# Hybrid Retrieval Systems

Hybrid retrieval combines dense vector retrieval and sparse lexical retrieval.
Dense retrieval represents text as numerical vectors. It can retrieve passages
that discuss related concepts even when they use different vocabulary.

Sparse retrieval uses lexical signals such as term frequency and inverse
document frequency. BM25 is a popular sparse retrieval algorithm. It is useful
when users search for technical identifiers, specific terminology, and exact
keywords.

# Reciprocal Rank Fusion

Reciprocal Rank Fusion combines multiple ranked result lists. It assigns
scores based on the position of each result in its original ranking.

A common RRF formula is the sum of one divided by a constant plus the rank.
This approach combines ranking signals without requiring the original
retrieval systems to produce comparable scores.

# Neural Reranking

A cross-encoder evaluates a query and candidate passage together. It produces
a relevance score for each query-passage pair.

A reranker can improve the ordering of retrieved candidates. It does not
replace retrieval because it can only reorder passages that were retrieved
in the first place.

# Retrieval Evaluation

Recall at K measures how many relevant documents appear in the first K results.
Mean Reciprocal Rank measures the position of the first relevant result.
Normalized Discounted Cumulative Gain rewards relevant results appearing
higher in a ranked list.

# Embedding Configuration

The embedding model used in this experiment is BAAI/bge-small-en-v1.5.
It produces 384-dimensional embeddings.
"""


TEST_QUERIES = [
    (
        "How does hybrid retrieval work?",
        QueryIntent.SEMANTIC,
    ),
    (
        "What is BM25?",
        QueryIntent.SEMANTIC,
    ),
    (
        "Compare dense retrieval and sparse retrieval.",
        QueryIntent.COMPARATIVE,
    ),
    (
        "What happens after retrieval and then reranking?",
        QueryIntent.MULTI_HOP,
    ),
    (
        "Find the exact model BAAI/bge-small-en-v1.5.",
        QueryIntent.EXACT,
    ),
]


def main() -> None:
    print("\n========== NORAG QUERY BEHAVIOUR TEST ==========\n")

    with TemporaryDirectory() as temp_dir:
        document_path = Path(temp_dir) / "retrieval_knowledge.md"
        document_path.write_text(SAMPLE_DOCUMENT, encoding="utf-8")

        print("[1/4] Ingesting and chunking document...")

        document = create_ingestion_service().ingest(document_path)
        chunks = StructuralChunker().chunk(document)

        if not chunks:
            raise RuntimeError("No chunks were created")

        print(f"Document elements: {len(document.elements)}")
        print(f"Chunks created: {len(chunks)}")

        print("\n[2/4] Building real retrieval indexes...")

        embedding_service = create_embedding_service(batch_size=8)
        embedded_chunks = embedding_service.embed_chunks(chunks)

        dense_index = InMemoryVectorIndex()
        sparse_index = BM25Index()

        dense_index.add_many(embedded_chunks)
        sparse_index.add_many(chunks)

        hybrid_retriever = HybridRetriever(
            embedding_service=embedding_service,
            dense_index=dense_index,
            sparse_index=sparse_index,
        )

        print(f"Dense index: {dense_index.count}")
        print(f"Sparse index: {sparse_index.count}")

        print("\n[3/4] Initializing adaptive retrieval...")

        service = create_adaptive_retrieval_service(
            classifier=QueryClassifier(),
            hybrid_retriever=hybrid_retriever,
            reranking_service=create_reranking_service(),
        )

        print("\n[4/4] Testing query scenarios...")

        passed = 0

        for number, (query, expected_intent) in enumerate(
            TEST_QUERIES,
            start=1,
        ):
            print(f"\n{'=' * 65}")
            print(f"TEST {number}: {query}")
            print(f"Expected intent: {expected_intent.value}")

            response = service.search(query=query, top_k=3)

            actual_intent = response.analysis.intent
            plan = response.plan

            print(f"Actual intent: {actual_intent.value}")
            print(f"Classifier route: {response.analysis.route.value}")
            print(f"Candidate count requested: {plan.candidate_k}")
            print(f"Dense weight: {plan.dense_weight}")
            print(f"Sparse weight: {plan.sparse_weight}")
            print(f"Results returned: {len(response.results)}")

            intent_passed = actual_intent == expected_intent
            retrieval_passed = bool(response.results)

            if intent_passed and retrieval_passed:
                passed += 1
                print("STATUS: PASS")
            else:
                print("STATUS: FAIL")

            for rank, item in enumerate(response.results, start=1):
                chunk = item.result.chunk

                print(f"\n  Result {rank}")
                print(f"  Score: {item.rerank_score:.4f}")
                print(f"  Section: {chunk.metadata.get('section_path')}")
                print(f"  Evidence: {chunk.text[:220]}...")

        print(f"\n{'=' * 65}")
        print(f"QUERY SCENARIOS PASSED: {passed}/{len(TEST_QUERIES)}")

        if passed != len(TEST_QUERIES):
            raise RuntimeError("Some query behaviour tests failed")

        print("\nNORAG QUERY BEHAVIOUR TEST COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
