from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    """A provider-independent LLM generation request."""

    system_prompt: str = ""
    user_prompt: str = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=512, ge=1)

    @field_validator("user_prompt")
    @classmethod
    def validate_user_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("User prompt cannot be empty")
        return value


class GenerationResponse(BaseModel):
    """Normalized response returned by an LLM provider."""

    content: str = Field(min_length=1)
    provider_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)

    finish_reason: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Generated content cannot be empty")
        return value
