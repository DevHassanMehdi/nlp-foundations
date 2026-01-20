# Assignment 2: Text Classification

This assignment builds a simple text classification pipeline using the NLTK movie_reviews dataset. It focuses on preprocessing, feature extraction, model training, and evaluation.

## Usage

From the repository root:

```bash
python -m assignments.A2_text_classification.src.run_all --help
python -m assignments.A2_text_classification.src.run_all
```

If NLTK resources are missing, run:

```bash
python -m nltk.downloader punkt punkt_tab movie_reviews
```

## Expected Outputs

Outputs are written to `assignments/A2_text_classification/outputs/`:

- `metrics.json` and `metrics.md` for evaluation results
- `vocab.json` for the learned vocabulary
