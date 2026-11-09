import threading
from uuid import uuid4

from norag.chunking.models import Chunk
from norag.embeddings.models import EmbeddedChunk
from norag.sparse_retrieval.bm25 import BM25Index
from norag.vector_store.memory import InMemoryVectorIndex


def run_concurrently(index, make_item, make_query) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                index.search(*make_query())
            except Exception as exc:  # noqa: BLE001
                errors.append((type(exc).__name__, str(exc)))
                stop.set()
                return

    def writer() -> None:
        for _ in range(500):
            if stop.is_set():
                return

            try:
                index.add_many([make_item()])
            except Exception as exc:  # noqa: BLE001
                errors.append((type(exc).__name__, str(exc)))
                stop.set()
                return

    readers = [threading.Thread(target=reader) for _ in range(3)]
    writer_thread = threading.Thread(target=writer)

    for thread in readers:
        thread.start()

    writer_thread.start()
    writer_thread.join(timeout=30)
    stop.set()

    for thread in readers:
        thread.join(timeout=5)

    return errors


def test_bm25_search_is_safe_during_index_mutation():
    document_id = uuid4()

    index = BM25Index()
    index.add_many(
        [
            Chunk(
                document_id=document_id,
                index=index_number,
                text=f"term{index_number} common retrieval words",
                token_count=4,
            )
            for index_number in range(300)
        ]
    )

    errors = run_concurrently(
        index,
        lambda: Chunk(
            document_id=document_id,
            index=uuid4().int % 100000,
            text="common retrieval words extra",
            token_count=4,
        ),
        lambda: ("common retrieval", 5),
    )

    assert errors == []
    assert index.count == 800


def test_dense_search_is_safe_during_index_mutation():
    document_id = uuid4()

    index = InMemoryVectorIndex()
    index.add_many(
        [
            EmbeddedChunk(
                chunk_id=uuid4(),
                document_id=document_id,
                index=index_number,
                text=f"chunk {index_number}",
                vector=[1.0, float(index_number % 7), 0.5],
                dimensions=3,
                model_name="test-model",
            )
            for index_number in range(300)
        ]
    )

    errors = run_concurrently(
        index,
        lambda: EmbeddedChunk(
            chunk_id=uuid4(),
            document_id=document_id,
            index=uuid4().int % 100000,
            text="new chunk",
            vector=[0.3, 0.2, 0.1],
            dimensions=3,
            model_name="test-model",
        ),
        lambda: ([1.0, 0.0, 0.5], 5),
    )

    assert errors == []
    assert index.count == 800
