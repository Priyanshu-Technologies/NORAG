import pytest
from pydantic import ValidationError

from norag.config import Settings


def test_default_verification_strategy():
    settings = Settings(_env_file=None)

    assert settings.verification_strategy == "lexical"


def test_ollama_verification_strategy():
    settings = Settings(
        _env_file=None,
        NORAG_VERIFICATION_STRATEGY="ollama",
    )

    assert settings.verification_strategy == "ollama"


def test_invalid_verification_strategy():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            NORAG_VERIFICATION_STRATEGY="invalid",
        )
