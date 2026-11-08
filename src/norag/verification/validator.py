from collections.abc import Sequence

from norag.prompting.models import PromptSource
from norag.verification.models import (
    CitationIssue,
    CitationStatus,
    CitationValidationReport,
    VerifiedSentence,
)
from norag.verification.parser import CitationParser


class CitationValidator:
    def __init__(self, parser: CitationParser | None = None) -> None:
        self.parser = parser or CitationParser()

    def validate(
        self,
        answer: str,
        sources: Sequence[PromptSource],
    ) -> CitationValidationReport:
        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        valid_source_numbers = {source.position for source in sources}

        sentences = self.parser.split_sentences(answer)

        verified_sentences: list[VerifiedSentence] = []
        issues: list[CitationIssue] = []

        cited_count = 0
        uncited_count = 0
        invalid_count = 0

        for index, sentence in enumerate(sentences, start=1):
            malformed = self.parser.extract_malformed_citations(sentence)
            malformed_set = set(malformed)

            citations = self.parser.extract_citations(sentence)
            cited_sources: list[int] = []
            sentence_has_invalid_citation = False

            for citation in citations:
                if not citation.isdigit():
                    label = f"[Source {citation}]"

                    if label in malformed_set:
                        continue

                    issues.append(
                        CitationIssue(
                            sentence_index=index,
                            citation=label,
                            reason="Source reference must contain a number.",
                        )
                    )
                    invalid_count += 1
                    sentence_has_invalid_citation = True
                    continue

                source_number = int(citation)

                if source_number not in valid_source_numbers:
                    issues.append(
                        CitationIssue(
                            sentence_index=index,
                            citation=f"[Source {source_number}]",
                            reason=(
                                "Referenced source does not exist "
                                "in the prompt source map."
                            ),
                        )
                    )
                    invalid_count += 1
                    sentence_has_invalid_citation = True
                    continue

                cited_sources.append(source_number)

            for citation in malformed:
                issues.append(
                    CitationIssue(
                        sentence_index=index,
                        citation=citation,
                        reason="Malformed citation syntax.",
                    )
                )
                invalid_count += 1
                sentence_has_invalid_citation = True

            if sentence_has_invalid_citation:
                status = CitationStatus.INVALID
            elif not citations:
                status = CitationStatus.UNCITED
                uncited_count += 1
            else:
                status = CitationStatus.VALID
                cited_count += 1

            verified_sentences.append(
                VerifiedSentence(
                    sentence_index=index,
                    text=sentence,
                    cited_sources=sorted(set(cited_sources)),
                    status=status,
                )
            )

        total = len(sentences)

        coverage = cited_count / total if total else 0.0

        return CitationValidationReport(
            total_sentences=total,
            cited_sentences=cited_count,
            uncited_sentences=uncited_count,
            invalid_citation_count=invalid_count,
            citation_coverage=coverage,
            sentences=verified_sentences,
            issues=issues,
        )
