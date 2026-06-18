from __future__ import annotations

import math

from hallucination_guard.models import ClaimVerification, SearchResult


class NLIVerifier:
    """Local NLI verifier.

    Uses a Hugging Face text-classification NLI model when available.
    The default model is lightweight enough for local experimentation, but you can
    change it later from config.

    If the model cannot be loaded, set fallback_to_lexical=True to use a simple
    lexical fallback. This keeps the pipeline runnable during early development.
    """

    def __init__(
        self,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        device: int = -1,
        fallback_to_lexical: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.fallback_to_lexical = fallback_to_lexical
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="text-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                return_all_scores=True,
                device=self.device,
            )
            return self._pipeline
        except Exception as exc:
            if not self.fallback_to_lexical:
                raise RuntimeError(f"Unable to load NLI model: {self.model_name}") from exc
            self._pipeline = False
            return self._pipeline

    def verify_claim(
        self,
        claim: str,
        evidence_results: list[SearchResult],
    ) -> ClaimVerification:
        if not evidence_results:
            return ClaimVerification(
                claim=claim,
                label="unsupported",
                entailment=0.0,
                contradiction=0.0,
                neutral=1.0,
            )

        pipe = self._load_pipeline()

        best: ClaimVerification | None = None

        for evidence in evidence_results:
            if pipe is False:
                verification = self._lexical_fallback(claim, evidence)
            else:
                verification = self._nli_score(claim, evidence)

            if best is None or self._rank_score(verification) > self._rank_score(best):
                best = verification

        assert best is not None
        return best

    def _nli_score(self, claim: str, evidence: SearchResult) -> ClaimVerification:
        premise = evidence.text
        hypothesis = claim
        model_input = f"{premise} </s></s> {hypothesis}"

        outputs = self._pipeline(model_input)[0]

        scores = {
            str(item["label"]).lower(): float(item["score"])
            for item in outputs
        }

        entailment = self._find_label_score(scores, "entail")
        contradiction = self._find_label_score(scores, "contrad")
        neutral = self._find_label_score(scores, "neutral")

        label = self._label_from_scores(entailment, contradiction, neutral)

        return ClaimVerification(
            claim=claim,
            label=label,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            evidence_doc_id=evidence.doc_id,
            evidence_text=evidence.text,
        )

    def _lexical_fallback(self, claim: str, evidence: SearchResult) -> ClaimVerification:
        claim_tokens = set(claim.lower().split())
        evidence_tokens = set(evidence.text.lower().split())

        if not claim_tokens:
            overlap = 0.0
        else:
            overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)

        # This is only a development fallback, not the research verifier.
        entailment = min(0.90, overlap)
        contradiction = 0.0
        neutral = max(0.0, 1.0 - entailment)

        label = "entailed" if entailment >= 0.60 else "unsupported"

        return ClaimVerification(
            claim=claim,
            label=label,
            entailment=entailment,
            contradiction=contradiction,
            neutral=neutral,
            evidence_doc_id=evidence.doc_id,
            evidence_text=evidence.text,
        )

    @staticmethod
    def _find_label_score(scores: dict[str, float], key: str) -> float:
        for label, score in scores.items():
            if key in label:
                return score
        return 0.0

    @staticmethod
    def _label_from_scores(entailment: float, contradiction: float, neutral: float) -> str:
        values = {
            "entailed": entailment,
            "contradicted": contradiction,
            "unsupported": neutral,
        }
        return max(values.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _rank_score(verification: ClaimVerification) -> float:
        # Prefer strong evidence: high entailment or high contradiction.
        return max(verification.entailment, verification.contradiction)
