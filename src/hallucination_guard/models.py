from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceDocument:
    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimVerification:
    claim: str
    label: str
    entailment: float
    contradiction: float
    neutral: float
    evidence_doc_id: str | None = None
    evidence_text: str | None = None


@dataclass
class GuardResult:
    question: str
    answer: str
    is_hallucinated: bool
    hallucination_probability: float
    unsupported_claim_ratio: float
    contradiction_claim_ratio: float
    claims: list[ClaimVerification]
    retrieved_evidence: list[SearchResult]
    corrected_answer: str | None = None
    action: str = "none"
