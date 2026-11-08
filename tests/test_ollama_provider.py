import json

import httpx
import pytest

from norag.generation.models import GenerationRequest
from norag.generation.providers.ollama import OllamaProvider


def create_provider(handler):
    transport = httpx.MockTransport(handler)

    return OllamaProvider(
        model_name="qwen3:1.7b",
        transport=transport,
        think=False,
    )


def test_provider_metadata():
    provider = OllamaProvider(model_name="qwen3:1.7b")

    assert provider.provider_name == "ollama"
    assert provider.model_name == "qwen3:1.7b"


def test_provider_generates_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3:1.7b",
                "message": {
                    "role": "assistant",
                    "content": "Hybrid retrieval combines dense and sparse search.",
                },
                "prompt_eval_count": 25,
                "eval_count": 12,
                "done_reason": "stop",
            },
        )

    provider = create_provider(handler)

    result = provider.generate(
        GenerationRequest(
            system_prompt="You are a helpful assistant.",
            user_prompt="Explain hybrid retrieval.",
        )
    )

    assert result.content == ("Hybrid retrieval combines dense and sparse search.")
    assert result.provider_name == "ollama"
    assert result.model_name == "qwen3:1.7b"
    assert result.input_tokens == 25
    assert result.output_tokens == 12
    assert result.finish_reason == "stop"


def test_provider_sends_correct_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)

        return httpx.Response(
            200,
            json={
                "model": "qwen3:1.7b",
                "message": {"content": "Hello!"},
            },
        )

    provider = create_provider(handler)

    provider.generate(
        GenerationRequest(
            system_prompt="Be concise.",
            user_prompt="Hello",
            temperature=0.5,
            max_output_tokens=128,
        )
    )

    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["payload"]["model"] == "qwen3:1.7b"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["options"]["temperature"] == 0.5
    assert captured["payload"]["options"]["num_predict"] == 128
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]


def test_provider_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal server error")

    provider = create_provider(handler)

    with pytest.raises(RuntimeError, match="Ollama generation request failed"):
        provider.generate(GenerationRequest(user_prompt="Hello"))


def test_provider_raises_on_empty_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen3:1.7b",
                "message": {"content": ""},
            },
        )

    provider = create_provider(handler)

    with pytest.raises(RuntimeError, match="empty or invalid content"):
        provider.generate(GenerationRequest(user_prompt="Hello"))


def test_provider_rejects_invalid_timeout():
    with pytest.raises(ValueError, match="Timeout"):
        OllamaProvider(
            model_name="qwen3:1.7b",
            timeout=0,
        )


def test_provider_rejects_empty_model_name():
    with pytest.raises(ValueError, match="Model name"):
        OllamaProvider(model_name=" ")
