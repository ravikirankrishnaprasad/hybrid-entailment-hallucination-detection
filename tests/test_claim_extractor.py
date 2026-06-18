from hallucination_guard.verification import ClaimExtractor


def test_extract_claims_from_sentences():
    extractor = ClaimExtractor()
    answer = "Bacterial pneumonia is treated with antibiotics. Oseltamivir is used for influenza."
    claims = extractor.extract(answer)

    assert len(claims) == 2
    assert "antibiotics" in claims[0]
    assert "influenza" in claims[1]
