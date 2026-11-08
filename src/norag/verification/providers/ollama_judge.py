import json
import re
from collections.abc import Sequence

from norag.generation.models import GenerationRequest
from norag.generation.service import GenerationService
from norag.verification.interfaces import EvidenceVerifier
from norag.verification.models import (
    ClaimEvidenceResult,
    EvidencePassage,
    EvidenceVerdict,
    ExtractedClaim,
)


class OllamaEvidenceVerifier(EvidenceVerifier):
    SYSTEM_PROMPT = """
You are an evidence verification system.

Evaluate the claim using ONLY the supplied evidence.

Choose exactly one verdict:

SUPPORTED:
At least one passage directly supports the claim.

CONTRADICTED:
At least one passage explicitly conflicts with the claim.

INSUFFICIENT_EVIDENCE:
The passages do not establish whether the claim is true or false.

Rules:
- Do not use outside knowledge.
- Do not assume missing information.
- Keyword overlap alone is not proof.
- Distinguish contradiction from missing evidence.
- Identify only the sources that directly support or contradict.
- Never invent source numbers.
- Return valid JSON only.
- Do not include Markdown fences.

Required JSON format:
{
    "verdict": "supported",
    "supporting_sources": [1],
    "contradicting_sources": [],
    "explanation": "The passage directly supports the claim."
}
"""

    def __init__(
        self,
        generation_service: GenerationService,
        max_output_tokens: int = 200,
    ) -> None:
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive.")

        self.generation_service = generation_service
        self.max_output_tokens = max_output_tokens

    def verify(
        self,
        claim: ExtractedClaim,
        evidence: Sequence[EvidencePassage],
    ) -> ClaimEvidenceResult:
        if not claim.cited_sources or not evidence:
            return ClaimEvidenceResult(
                claim=claim,
                verdict=EvidenceVerdict.INSUFFICIENT_EVIDENCE,
                explanation="No cited evidence is available.",
                method="ollama_llm_judge",
            )

        allowed_sources = {passage.source_number for passage in evidence}

        evidence_text = "\n\n".join(
            (
                f"[Source {passage.source_number}]\n"
                f"Section: {passage.section or 'Unknown'}\n"
                f"Content: {passage.text}"
            )
            for passage in evidence
        )

        user_prompt = (
            f"CLAIM:\n{claim.text}\n\n"
            f"EVIDENCE:\n{evidence_text}\n\n"
            "Determine the relationship between the claim and evidence."
        )

        response = self.generation_service.generate(
            GenerationRequest(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            )
        )

        result = self._parse_response(
            response.content,
            allowed_sources=allowed_sources,
        )

        return ClaimEvidenceResult(
            claim=claim,
            verdict=result["verdict"],
            supporting_sources=result["supporting_sources"],
            contradicting_sources=result["contradicting_sources"],
            explanation=result["explanation"],
            method="ollama_llm_judge",
        )

    @staticmethod
    def _parse_response(
        content: str,
        allowed_sources: set[int],
    ) -> dict:
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content.strip(),
            flags=re.IGNORECASE,
        )

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Ollama returned invalid verification JSON.") from exc

        if not isinstance(data, dict):
            raise TypeError("Verification response must be a JSON object.")

        try:
            verdict = EvidenceVerdict(data["verdict"])
            supporting = data["supporting_sources"]
            contradicting = data["contradicting_sources"]
            explanation = data["explanation"]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Ollama returned an invalid verification verdict."
            ) from exc

        if not isinstance(supporting, list) or not isinstance(contradicting, list):
            raise TypeError("Source attributions must be lists.")

        for source_list in (supporting, contradicting):
            if any(
                not isinstance(source, int) or isinstance(source, bool)
                for source in source_list
            ):
                raise ValueError("Source references must be integers.")

            if len(source_list) != len(set(source_list)):
                raise ValueError("Duplicate source references are not allowed.")

            if not set(source_list).issubset(allowed_sources):
                raise ValueError("Ollama referenced a source that was not supplied.")

        if set(supporting) & set(contradicting):
            raise ValueError("A source cannot be both supporting and contradicting.")

        if verdict == EvidenceVerdict.SUPPORTED:
            if not supporting or contradicting:
                raise ValueError("Supported verdict requires supporting sources only.")

        elif verdict == EvidenceVerdict.CONTRADICTED:
            if not contradicting or supporting:
                raise ValueError(
                    "Contradicted verdict requires contradicting sources only."
                )

        elif supporting or contradicting:
            raise ValueError(
                "Insufficient evidence verdict cannot contain source attributions."
            )

        if not isinstance(explanation, str) or not explanation.strip():
            raise ValueError("Verification explanation cannot be empty.")

        return {
            "verdict": verdict,
            "supporting_sources": supporting,
            "contradicting_sources": contradicting,
            "explanation": explanation.strip(),
        }
