import re


class ClaimExtractor:
    """Rule-based atomic claim extractor.

    For the MVP, each sentence is treated as a claim.
    Later, replace this with dependency parsing or an LLM-free factual claim splitter.
    """

    _sentence_pattern = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, min_words: int = 4) -> None:
        self.min_words = min_words

    def extract(self, answer: str) -> list[str]:
        if not answer or not answer.strip():
            return []

        raw_sentences = self._sentence_pattern.split(answer.strip())
        claims: list[str] = []

        for sentence in raw_sentences:
            claim = sentence.strip()
            if not claim:
                continue

            word_count = len(re.findall(r"\w+", claim))
            if word_count >= self.min_words:
                claims.append(claim)

        if not claims and answer.strip():
            claims.append(answer.strip())

        return claims
