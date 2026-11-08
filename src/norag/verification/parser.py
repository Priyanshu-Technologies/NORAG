import re


class CitationParser:
    CITATION_PATTERN = re.compile(
        r"\[Source\s+([^\]]+)\]",
        re.IGNORECASE,
    )

    MALFORMED_CITATION_PATTERN = re.compile(
        r"\[Source\b[^\]]*\]",
        re.IGNORECASE,
    )

    SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

    def split_sentences(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        paragraphs = [
            paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()
        ]

        sentences: list[str] = []

        for paragraph in paragraphs:
            if paragraph.startswith("#"):
                continue

            parts = self.SENTENCE_PATTERN.split(paragraph)

            sentences.extend(part.strip() for part in parts if part.strip())

        return sentences

    def extract_citations(self, text: str) -> list[str]:
        return [
            match.group(1).strip() for match in self.CITATION_PATTERN.finditer(text)
        ]

    def extract_malformed_citations(self, text: str) -> list[str]:
        malformed: list[str] = []

        for match in self.MALFORMED_CITATION_PATTERN.finditer(text):
            content = match.group(0)
            citation = self.CITATION_PATTERN.fullmatch(content)

            if citation is None:
                malformed.append(content)
                continue

            source_number = citation.group(1).strip()

            if not source_number.isdigit():
                malformed.append(content)

        return malformed
