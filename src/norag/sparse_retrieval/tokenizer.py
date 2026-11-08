import re
import unicodedata

TOKEN_PATTERN = re.compile(r"(?u)[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Normalize text and split it into searchable terms."""

    normalized = unicodedata.normalize("NFKC", text).casefold()

    return TOKEN_PATTERN.findall(normalized)
