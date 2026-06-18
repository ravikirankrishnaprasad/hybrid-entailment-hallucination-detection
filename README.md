# Hybrid Entailment Hallucination Detection

Work-in-progress research implementation for local hallucination detection and evidence-grounded verification.

## Status

Initial development.

## Goal

Build a local and reproducible framework for detecting unsupported or contradicted claims in generated responses using retrieval and verification techniques.

## Planned Components

- Retrieval module
- Evidence reranking module
- Claim verification module
- Hallucination classification module
- Gated correction module
- Evaluation scripts

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_demo.py
```

## Notes

This repository is under active development.

Implementation details, experiments, dataset preparation, and documentation will be updated as the project matures.
