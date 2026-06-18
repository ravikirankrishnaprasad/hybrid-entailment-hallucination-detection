from hallucination_guard.models import EvidenceDocument
from hallucination_guard.retrieval import BM25Retriever


def test_bm25_returns_relevant_document():
    docs = [
        EvidenceDocument(doc_id="1", text="Bacterial pneumonia is treated with antibiotics."),
        EvidenceDocument(doc_id="2", text="Stars are visible in the night sky."),
    ]

    retriever = BM25Retriever().fit(docs)
    results = retriever.search("pneumonia treatment", top_k=1)

    assert results[0].doc_id == "1"
