from __future__ import annotations

from hallucination_guard.correction import GatedCorrector
from hallucination_guard.models import EvidenceDocument, GuardResult
from hallucination_guard.retrieval import BM25Retriever
from hallucination_guard.verification import ClaimExtractor, NLIVerifier


class HallucinationGuard:
    """End-to-end local hallucination detection pipeline.

    MVP pipeline:
    question + answer
    -> BM25 evidence retrieval
    -> sentence-level claim extraction
    -> NLI verification
    -> hallucination scoring
    -> gated correction
    """

    def __init__(
        self,
        retriever: BM25Retriever | None = None,
        claim_extractor: ClaimExtractor | None = None,
        verifier: NLIVerifier | None = None,
        corrector: GatedCorrector | None = None,
        entailment_threshold: float = 0.55,
        contradiction_threshold: float = 0.55,
        unsupported_threshold: float = 0.50,
    ) -> None:
        self.retriever = retriever or BM25Retriever()
        self.claim_extractor = claim_extractor or ClaimExtractor()
        self.verifier = verifier or NLIVerifier()
        self.corrector = corrector or GatedCorrector()
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.unsupported_threshold = unsupported_threshold

    def fit(self, documents: list[EvidenceDocument]) -> "HallucinationGuard":
        self.retriever.fit(documents)
        return self

    def analyze(self, question: str, answer: str, top_k: int = 5) -> GuardResult:
        retrieved = self.retriever.search(question, top_k=top_k)
        claims = self.claim_extractor.extract(answer)

        claim_verifications = [
            self.verifier.verify_claim(claim, retrieved)
            for claim in claims
        ]

        unsupported_count = 0
        contradiction_count = 0

        for item in claim_verifications:
            is_entailed = item.entailment >= self.entailment_threshold
            is_contradicted = item.contradiction >= self.contradiction_threshold

            if is_contradicted:
                contradiction_count += 1
            elif not is_entailed:
                unsupported_count += 1

        claim_count = max(1, len(claim_verifications))
        unsupported_ratio = unsupported_count / claim_count
        contradiction_ratio = contradiction_count / claim_count

        hallucination_probability = min(
            1.0,
            (0.65 * unsupported_ratio) + (0.35 * contradiction_ratio),
        )

        is_hallucinated = (
            contradiction_ratio > 0.0
            or unsupported_ratio >= self.unsupported_threshold
        )

        corrected_answer = self.corrector.correct(
            original_answer=answer,
            hallucination_probability=hallucination_probability,
            claim_verifications=claim_verifications,
        )

        action = "corrected" if corrected_answer else "no_change"

        return GuardResult(
            question=question,
            answer=answer,
            is_hallucinated=is_hallucinated,
            hallucination_probability=hallucination_probability,
            unsupported_claim_ratio=unsupported_ratio,
            contradiction_claim_ratio=contradiction_ratio,
            claims=claim_verifications,
            retrieved_evidence=retrieved,
            corrected_answer=corrected_answer,
            action=action,
        )
