from hallucination_guard import HallucinationGuard
from hallucination_guard.models import EvidenceDocument


def main() -> None:
    documents = [
        EvidenceDocument(
            doc_id="med_001",
            text="Bacterial pneumonia is commonly treated with antibiotics such as amoxicillin or azithromycin.",
            metadata={"source": "demo"},
        ),
        EvidenceDocument(
            doc_id="med_002",
            text="Oseltamivir is an antiviral medication used for influenza treatment.",
            metadata={"source": "demo"},
        ),
        EvidenceDocument(
            doc_id="med_003",
            text="Viral infections are not treated with antibiotics unless there is a bacterial co-infection.",
            metadata={"source": "demo"},
        ),
    ]

    question = "What is the primary treatment for bacterial pneumonia?"
    answer = "Bacterial pneumonia is primarily treated using antiviral medications such as oseltamivir."

    guard = HallucinationGuard().fit(documents)
    result = guard.analyze(question=question, answer=answer, top_k=3)

    print("Question:", result.question)
    print("Answer:", result.answer)
    print("Is hallucinated:", result.is_hallucinated)
    print("Hallucination probability:", round(result.hallucination_probability, 3))
    print("Action:", result.action)

    print("\nClaims:")
    for claim in result.claims:
        print("-", claim.claim)
        print("  label:", claim.label)
        print("  entailment:", round(claim.entailment, 3))
        print("  contradiction:", round(claim.contradiction, 3))
        print("  neutral:", round(claim.neutral, 3))
        print("  evidence:", claim.evidence_text)

    if result.corrected_answer:
        print("\nCorrected answer:", result.corrected_answer)


if __name__ == "__main__":
    main()
