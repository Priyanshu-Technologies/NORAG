from pathlib import Path
from tempfile import TemporaryDirectory

from norag.chunking.strategies.structural import StructuralChunker
from norag.embeddings.factory import create_embedding_service
from norag.hybrid_retrieval.service import HybridRetriever
from norag.ingestion.factory import create_ingestion_service
from norag.query_understanding.classifier import QueryClassifier
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

Evaluation datasets should contain queries and manually verified relevant
passages. Retrieval quality should be measured rather than assumed.
"""


def main() -> None:
    print("\nNORAG END-TO-END TEST\n")

    with TemporaryDirectory() as temp_dir:
        document_path = Path(temp_dir) / "hybrid_retrieval.md"
        document_path.write_text(SAMPLE_DOCUMENT, encoding="utf-8")

        # 1. Document ingestion
        print("[1/7] Ingesting document...")

        ingestion_service = create_ingestion_service()
        document = ingestion_service.ingest(document_path)

        print(f"Document ID: {document.id}")
        print(f"Extracted elements: {len(document.elements)}")

        # 2. Structural chunking
        print("\n[2/7] Creating structural chunks...")

        chunker = StructuralChunker()
        chunks = chunker.chunk(document)

        if not chunks:
            raise RuntimeError("Ingestion produced no chunks")

        print(f"Created chunks: {len(chunks)}")

        # 3. Generate embeddings
        print("\n[3/7] Generating embeddings...")

        embedding_service = create_embedding_service(
            batch_size=8,
        )

        embedded_chunks = embedding_service.embed_chunks(chunks)

        if len(embedded_chunks) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        print(f"Embedded chunks: {len(embedded_chunks)}")
        print(f"Embedding dimensions: {embedded_chunks[0].dimensions}")

        # 4. Populate dense and sparse indexes
        print("\n[4/7] Building retrieval indexes...")

        dense_index = InMemoryVectorIndex()
        sparse_index = BM25Index()

        dense_index.add_many(embedded_chunks)
        sparse_index.add_many(chunks)

        print(f"Dense index count: {dense_index.count}")
        print(f"Sparse index count: {sparse_index.count}")

        # 5. Initialize hybrid retrieval and reranking
        print("\n[5/7] Initializing retrieval components...")

        hybrid_retriever = HybridRetriever(
            embedding_service=embedding_service,
            dense_index=dense_index,
            sparse_index=sparse_index,
        )

        classifier = QueryClassifier()
        reranking_service = create_reranking_service()

        adaptive_service = create_adaptive_retrieval_service(
            classifier=classifier,
            hybrid_retriever=hybrid_retriever,
            reranking_service=reranking_service,
        )

        # 6. Execute a real query
        query = (
            "How does hybrid retrieval combine semantic similarity "
            "with keyword matching?"
        )

        print("\n[6/7] Executing adaptive retrieval...")
        print(f"Query: {query}")

        response = adaptive_service.search(
            query=query,
            top_k=3,
        )

        print(f"\nDetected intent: {response.analysis.intent}")
        print(f"Selected route: {response.analysis.route}")
        print(f"Classification rule: {response.analysis.rule}")

        print("\nAdaptive retrieval plan:")
        print(f"  Final top K: {response.plan.final_top_k}")
        print(f"  Candidate K: {response.plan.candidate_k}")
        print(f"  Dense weight: {response.plan.dense_weight}")
        print(f"  Sparse weight: {response.plan.sparse_weight}")

        # 7. Inspect actual ranked evidence
        print("\n[7/7] Final reranked evidence...")

        if not response.results:
            raise RuntimeError("NORAG returned no retrieval results")

        for rank, item in enumerate(response.results, start=1):
            chunk = item.result.chunk

            print(f"\n--- RESULT {rank} ---")
            print(f"Chunk ID: {chunk.id}")
            print(f"Rerank score: {item.rerank_score:.5f}")
            print(f"Original retrieval rank: {item.original_rank}")
            print(f"Source element IDs: {chunk.source_element_ids}")
            print(f"Source locations: {chunk.source_locations}")
            print(f"Section: {chunk.metadata.get('section_path')}")
            print(f"Text:\n{chunk.text[:700]}")

        print("\nNORAG TEST COMPLETED\n")


if __name__ == "__main__":
    main()
