from norag.generation.models import GenerationRequest
from norag.generation.providers.ollama import OllamaProvider
from norag.generation.service import GenerationService

provider = OllamaProvider(
    model_name="qwen3:1.7b",
)

generation_service = GenerationService(provider)

response = generation_service.generate(
    GenerationRequest(
        system_prompt=(
            "You are NORAG, a technical research assistant. "
            "Give clear, accurate and concise answers."
        ),
        user_prompt=(
            "/no_think\n"
            "Explain the difference between dense retrieval and BM25 "
            "in three concise points."
        ),
        temperature=0.2,
        max_output_tokens=512,
    )
)

print("\n--- NORAG GENERATION TEST ---")
print(f"Provider: {response.provider_name}")
print(f"Model: {response.model_name}")
print(f"Input tokens: {response.input_tokens}")
print(f"Output tokens: {response.output_tokens}")
print(f"\nResponse:\n{response.content}")
