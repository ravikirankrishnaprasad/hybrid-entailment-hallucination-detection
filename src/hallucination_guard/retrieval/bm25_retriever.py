import re
from typing import Iterable

from rank_bm25 import BM25Okapi

from hallucination_guard.models import EvidenceDocument, SearchResult


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Simple local BM25 retriever.

    This is intentionally lightweight for the first implementation.
    Later, this can be combined with dense retrieval in a HybridRetriever.
    """

    def __init__(self) -> None:
        self.documents: list[EvidenceDocument] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

    def fit(self, documents: Iterable[EvidenceDocument]) -> "BM25Retriever":
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("BM25Retriever.fit() requires at least one document.")

        self._tokenized_corpus = [tokenize(doc.text) for doc in self.documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        return self

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever is not fitted. Call fit() first.")

        tokenized_query = tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda idx: float(scores[idx]),
            reverse=True,
        )[:top_k]

        results: list[SearchResult] = []
        for idx in ranked_indices:
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    text=doc.text,
                    score=float(scores[idx]),
                    metadata=doc.metadata,
                )
            )
        return results
