import httpx

from norag.generation.interfaces import GenerationProvider
from norag.generation.models import GenerationRequest, GenerationResponse


class OllamaProvider(GenerationProvider):
    """Generate responses using a locally running Ollama instance."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        think: bool = False,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._transport = transport
        self._think = think

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages = []

        if request.system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": request.system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": request.user_prompt,
            }
        )

        payload = {
            "model": self._model_name,
            "messages": messages,
            "stream": False,
            "think": self._think,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }

        try:
            with httpx.Client(
                timeout=self._timeout,
                transport=self._transport,
            ) as client:
                response = client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama generation request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise TypeError("Ollama returned an invalid response.")

        message = data.get("message")

        if not isinstance(message, dict):
            raise TypeError("Ollama response is missing its message.")

        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned empty or invalid content.")

        return GenerationResponse(
            content=content,
            provider_name=self.provider_name,
            model_name=data.get("model", self._model_name),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            finish_reason=data.get("done_reason"),
        )
