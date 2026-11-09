from pathlib import Path
from tempfile import TemporaryDirectory

from norag.answering.service import AnsweringService
from norag.chunking.strategies.structural import StructuralChunker
from norag.context.assembler import ContextAssembler
from norag.embeddings.factory import create_embedding_service
from norag.generation.providers.ollama import OllamaProvider
from norag.generation.service import GenerationService
from norag.hybrid_retrieval.service import HybridRetriever
from norag.ingestion.factory import create_ingestion_service
from norag.prompting.builder import PromptBuilder
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
    print("              NORAG END-TO-END GENERATION TEST")

    with TemporaryDirectory() as temp_dir:
        document_path = Path(temp_dir) / "hybrid_retrieval.md"
        document_path.write_text(SAMPLE_DOCUMENT, encoding="utf-8")

        # 1. Document ingestion
        print("\n[1/8] Ingesting document...")

        ingestion_service = create_ingestion_service()
        document = ingestion_service.ingest(document_path)

        print(f"Document ID: {document.id}")
        print(f"Extracted elements: {len(document.elements)}")

        # 2. Structural chunking
        print("\n[2/8] Creating structural chunks...")

        chunker = StructuralChunker()
        chunks = chunker.chunk(document)

        if not chunks:
            raise RuntimeError("Ingestion produced no chunks")

        print(f"Created chunks: {len(chunks)}")

        # 3. Generate embeddings
        print("\n[3/8] Generating embeddings...")

        embedding_service = create_embedding_service(
            batch_size=8,
        )

        embedded_chunks = embedding_service.embed_chunks(chunks)

        if len(embedded_chunks) != len(chunks):
            raise RuntimeError("Embedding count does not match chunk count")

        print(f"Embedded chunks: {len(embedded_chunks)}")
        print(f"Dimensions: {embedded_chunks[0].dimensions}")

        # 4. Build dense and sparse indexes
        print("\n[4/8] Building retrieval indexes...")

        dense_index = InMemoryVectorIndex()
        sparse_index = BM25Index()

        dense_index.add_many(embedded_chunks)
        sparse_index.add_many(chunks)

        print(f"Dense index count: {dense_index.count}")
        print(f"Sparse index count: {sparse_index.count}")

        # 5. Initialize adaptive retrieval
        print("\n[5/8] Initializing adaptive retrieval...")

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

        # 6. Initialize context, prompting and generation
        print("\n[6/8] Initializing answering pipeline...")

        context_assembler = ContextAssembler()
        prompt_builder = PromptBuilder()

        ollama_provider = OllamaProvider(
            model_name="qwen3:1.7b",
            think=False,
            timeout=180.0,
        )

        generation_service = GenerationService(
            provider=ollama_provider,
        )

        answering_service = AnsweringService(
            retrieval_service=adaptive_service,
            context_assembler=context_assembler,
            prompt_builder=prompt_builder,
            generation_service=generation_service,
        )

        print("Retrieval service: Ready")
        print("Context assembler: Ready")
        print("Prompt builder: Ready")
        print("Generation provider: Ollama / qwen3:1.7b")

        # 7. Ask NORAG a question
        query = (
            "How does hybrid retrieval combine semantic similarity "
            "with keyword matching?"
        )

        print("\n[7/8] Sending query through NORAG...")
        print(f"\nQuestion: {query}")

        response = answering_service.answer(
            query=query,
            top_k=3,
        )

        if not response.retrieval.results:
            raise RuntimeError("NORAG retrieved no relevant results")

        if not response.context.items:
            raise RuntimeError("NORAG assembled no evidence")

        # 8. Inspect the complete response
        print("\n[8/8] NORAG generated its answer!")

        print("GENERATED ANSWER")

        print(f"\n{response.answer}")

        print("RETRIEVAL TRACE")

        print(f"Intent: {response.retrieval.analysis.intent}")
        print(f"Route: {response.retrieval.analysis.route}")
        print(f"Rule: {response.retrieval.analysis.rule}")
        print(f"Candidate K: {response.retrieval.plan.candidate_k}")
        print(f"Dense weight: {response.retrieval.plan.dense_weight}")
        print(f"Sparse weight: {response.retrieval.plan.sparse_weight}")

        print("ASSEMBLED EVIDENCE")

        print(f"Included passages: {response.context.included_count}")
        print(f"Excluded passages: {response.context.excluded_count}")
        print(f"Estimated context tokens: {response.context.token_count}")

        for item in response.context.items:
            chunk = item.result.result.chunk

            print(f"\n[Source {item.position}]")
            print(f"Section: {chunk.metadata.get('section_path')}")
            print(f"Rerank score: {item.result.rerank_score:.5f}")
            print(f"Content:\n{chunk.text[:500]}")

        print("SOURCE MAPPING")

        for source in response.prompt.sources:
            print(f"\n{source.source_id}")
            print(f"Document: {source.document_id}")
            print(f"Chunk: {source.chunk_id}")
            print(f"Section: {source.section}")
            print(f"Pages: {source.page_numbers}")

        print("GENERATION METRICS")

        print(f"Provider: {response.generation.provider_name}")
        print(f"Model: {response.generation.model_name}")
        print(f"Input tokens: {response.generation.input_tokens}")
        print(f"Output tokens: {response.generation.output_tokens}")
        print(f"Finish reason: {response.generation.finish_reason}")
        print(f"Estimated prompt tokens: {response.prompt.estimated_input_tokens}")

        print("NORAG END-TO-END TEST COMPLETED")


if __name__ == "__main__":
    main()
