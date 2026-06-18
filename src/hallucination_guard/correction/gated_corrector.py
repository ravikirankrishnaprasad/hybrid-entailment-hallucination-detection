from hallucination_guard.models import ClaimVerification


class GatedCorrector:
    """Simple gated correction module.

    MVP behavior:
    - If evidence is strong and hallucination confidence is high, return evidence-based answer.
    - Otherwise return None, meaning "do not modify".

    This avoids the V1 problem where factual answers may be unnecessarily rewritten.
    """

    def __init__(
        self,
        min_hallucination_probability: float = 0.70,
        min_contradiction: float = 0.60,
        min_entailment_for_replacement: float = 0.55,
    ) -> None:
        self.min_hallucination_probability = min_hallucination_probability
        self.min_contradiction = min_contradiction
        self.min_entailment_for_replacement = min_entailment_for_replacement

    def correct(
        self,
        original_answer: str,
        hallucination_probability: float,
        claim_verifications: list[ClaimVerification],
    ) -> str | None:
        if hallucination_probability < self.min_hallucination_probability:
            return None

        corrected_parts: list[str] = []

        for item in claim_verifications:
            if item.label == "entailed":
                corrected_parts.append(item.claim)
                continue

            if (
                item.label == "contradicted"
                and item.contradiction >= self.min_contradiction
                and item.evidence_text
            ):
                corrected_parts.append(item.evidence_text)
                continue

            # Unsupported claims are removed in the MVP.
            # Later, replace this with evidence-aware rewriting.
            continue

        corrected = " ".join(dict.fromkeys(corrected_parts)).strip()
        if not corrected:
            return None

        if corrected == original_answer.strip():
            return None

        return corrected
