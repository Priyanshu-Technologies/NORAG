from norag.answering.models import AnswerRequest
from norag.answering.service import AnsweringService
from norag.context.models import AssembledContext
from norag.generation.models import GenerationRequest, GenerationResponse
from norag.prompting.models import PromptConstruction
from norag.query_understanding.models import (
    QueryAnalysis,
    QueryIntent,
    RetrievalRoute,
)
from norag.retrieval_policy.models import RetrievalPlan
from norag.retrieval_policy.service import AdaptiveRetrievalResponse


def create_pipeline_objects():
    retrieval = AdaptiveRetrievalResponse(
        analysis=QueryAnalysis(
            query="Explain hybrid retrieval.",
            intent=QueryIntent.SEMANTIC,
            route=RetrievalRoute.DENSE,
            dense_weight=1.0,
            sparse_weight=0.0,
            rule="semantic_fallback",
        ),
        plan=RetrievalPlan(
            intent=QueryIntent.SEMANTIC,
            final_top_k=5,
            candidate_k=15,
            dense_weight=0.8,
            sparse_weight=0.2,
        ),
        results=[],
    )

    context = AssembledContext(
        items=[],
        context_text="Retrieved technical evidence.",
        token_count=4,
        available_tokens=100,
        included_count=0,
        excluded_count=0,
    )

    generation_request = GenerationRequest(
        user_prompt="Explain hybrid retrieval.",
    )

    prompt = PromptConstruction(
        request=generation_request,
        sources=[],
        estimated_input_tokens=20,
    )

    generation = GenerationResponse(
        content="Hybrid retrieval combines dense and sparse search.",
        provider_name="ollama",
        model_name="qwen3:1.7b",
        input_tokens=20,
        output_tokens=12,
        finish_reason="stop",
    )

    return retrieval, context, prompt, generation


class FakeRetrievalService:
    def __init__(self, retrieval, events):
        self.retrieval = retrieval
        self.events = events
        self.calls = []

    def search(self, query, top_k):
        self.events.append("retrieval")
        self.calls.append((query, top_k))
        return self.retrieval


class FakeContextAssembler:
    def __init__(self, context, events):
        self.context = context
        self.events = events
        self.received_results = None

    def assemble(self, results):
        self.events.append("context")
        self.received_results = results
        return self.context


class FakePromptBuilder:
    def __init__(self, prompt, events):
        self.prompt = prompt
        self.events = events
        self.received_query = None
        self.received_context = None

    def build(self, query, context):
        self.events.append("prompt")
        self.received_query = query
        self.received_context = context
        return self.prompt


class FakeGenerationService:
    def __init__(self, generation, events):
        self.generation = generation
        self.events = events
        self.received_request = None

    def generate(self, request):
        self.events.append("generation")
        self.received_request = request
        return self.generation


def create_service():
    retrieval, context, prompt, generation = create_pipeline_objects()
    events = []

    retrieval_service = FakeRetrievalService(retrieval, events)
    context_assembler = FakeContextAssembler(context, events)
    prompt_builder = FakePromptBuilder(prompt, events)
    generation_service = FakeGenerationService(generation, events)

    service = AnsweringService(
        retrieval_service=retrieval_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        generation_service=generation_service,
    )

    return (
        service,
        retrieval_service,
        context_assembler,
        prompt_builder,
        generation_service,
        events,
    )


def test_answering_service_generates_answer():
    service, *_ = create_service()

    response = service.answer("Explain hybrid retrieval.")

    assert response.answer == ("Hybrid retrieval combines dense and sparse search.")
    assert response.generation.provider_name == "ollama"
    assert response.generation.model_name == "qwen3:1.7b"


def test_answering_service_preserves_query():
    service, *_ = create_service()

    response = service.answer("  Explain hybrid retrieval.  ")

    assert response.query == "Explain hybrid retrieval."


def test_answering_service_forwards_top_k():
    service, retrieval_service, *_ = create_service()

    service.answer("Explain hybrid retrieval.", top_k=8)

    assert retrieval_service.calls == [("Explain hybrid retrieval.", 8)]


def test_answering_service_passes_retrieval_results_to_context():
    service, _, context_assembler, *_ = create_service()

    service.answer("Explain hybrid retrieval.")

    assert context_assembler.received_results == []


def test_answering_service_passes_context_to_prompt_builder():
    service, _, context_assembler, prompt_builder, *_ = create_service()

    service.answer("Explain hybrid retrieval.")

    assert prompt_builder.received_context is context_assembler.context
    assert prompt_builder.received_query == "Explain hybrid retrieval."


def test_answering_service_passes_prompt_to_generation():
    service, _, _, prompt_builder, generation_service, *_ = create_service()

    service.answer("Explain hybrid retrieval.")

    assert generation_service.received_request is prompt_builder.prompt.request


def test_answering_service_executes_stages_in_order():
    service, *remaining = create_service()
    events = remaining[-1]

    service.answer("Explain hybrid retrieval.")

    assert events == [
        "retrieval",
        "context",
        "prompt",
        "generation",
    ]


def test_answering_service_rejects_empty_query():
    service, *_ = create_service()

    try:
        service.answer("   ")
    except ValueError as exc:
        assert "Query cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_answering_service_rejects_invalid_top_k():
    service, *_ = create_service()

    try:
        service.answer("Explain hybrid retrieval.", top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_answer_request_strips_query():
    request = AnswerRequest(query="  Explain retrieval.  ")

    assert request.query == "Explain retrieval."


def test_answering_service_propagates_generation_errors():
    service, _, _, _, generation_service, _ = create_service()

    def failing_generate(request):
        raise RuntimeError("Model unavailable")

    generation_service.generate = failing_generate

    try:
        service.answer("Explain hybrid retrieval.")
    except RuntimeError as exc:
        assert "Model unavailable" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError")


def test_answering_service_includes_citation_validation():
    service, *_ = create_service()

    response = service.answer("Explain hybrid retrieval.")

    assert response.citation_validation is not None
    assert response.citation_validation.total_sentences == 1
    assert response.citation_validation.cited_sentences == 0
    assert response.citation_validation.uncited_sentences == 1
    assert response.citation_validation.citation_coverage == 0.0
    assert response.citation_validation.citations_are_valid


def test_answering_service_includes_claim_verification():
    service, *_ = create_service()

    response = service.answer("Explain hybrid retrieval.")

    report = response.claim_verification

    assert report is not None
    assert report.total_claims == 1
    assert report.supported_claims == 0
    assert report.contradicted_claims == 0
    assert report.insufficient_claims == 1
    assert report.has_unsupported_claims
