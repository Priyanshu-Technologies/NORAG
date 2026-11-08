import re

from norag.verification.models import ExtractedClaim
from norag.verification.parser import CitationParser


class ClaimExtractor:
    CITATION_PATTERN = re.compile(
        r"\[Source\s+\d+\]",
        re.IGNORECASE,
    )

    def __init__(
        self,
        parser: CitationParser | None = None,
    ) -> None:
        self.parser = parser or CitationParser()

    def extract(self, answer: str) -> list[ExtractedClaim]:
        if not answer or not answer.strip():
            return []

        claims: list[ExtractedClaim] = []

        for index, sentence in enumerate(
            self.parser.split_sentences(answer),
            start=1,
        ):
            citations = self.parser.extract_citations(sentence)

            claim_text = self.CITATION_PATTERN.sub(
                "",
                sentence,
            ).strip()

            if not claim_text:
                continue

            claims.append(
                ExtractedClaim(
                    sentence_index=index,
                    text=claim_text,
                    cited_sources=sorted(
                        {int(citation) for citation in citations if citation.isdigit()}
                    ),
                )
            )

        return claims
