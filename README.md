# TIES4200 NLP Course Repository

Author: Hassan Mehdi

This repository contains all assignments and the main project for TIES4200 (Independent NLP). Assignment 1 maps to Jurafsky & Martin Chapters 2–3 by implementing tokenization, n-gram language models, and evaluation with cross-entropy and perplexity.

## Setup

### Option A: Conda (recommended)

```bash
./setup_conda.sh
python -m nltk.downloader punkt punkt_tab gutenberg
```

### Option B: venv

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies and NLTK resources:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab gutenberg
```

## Run Assignment 1

```bash
python -m assignments.A1_foundations.src.run_all --help
python -m assignments.A1_foundations.src.run_all
```

Outputs are written to `assignments/A1_foundations/outputs/`.

## Run Assignment 2

```bash
python -m nltk.downloader punkt punkt_tab movie_reviews
python -m assignments.A2_text_classification.src.run_all --help
python -m assignments.A2_text_classification.src.run_all
```

Outputs are written to `assignments/A2_text_classification/outputs/`.

## Assignment 1 Code Overview

Key modules for Assignment 1 live under `assignments/A1_foundations/src/`:

- `load_data.py`: Loads the Gutenberg text (`austen-emma.txt`) with helpful errors if NLTK data is missing.
- `tokenize_stats.py`: Tokenizes, lowercases, removes punctuation-only tokens, and computes corpus stats.
- `ngram_lm.py`: Implements UnigramLM and BigramLM, `<UNK>` replacement, `<s>` starts, and add-k smoothing.
- `evaluate.py`: Train/test split by sentence, cross-entropy, and perplexity calculations.
- `utils.py`: Reproducibility and IO helpers (seed, safe log, JSON/Markdown saves).
- `run_all.py`: Orchestrates the full pipeline, saves outputs, and prints a summary.

## Pipeline Summary

1) Load raw text from NLTK Gutenberg.
2) Tokenize, lowercase, and keep sentence boundaries.
3) Optionally truncate tokens for quick runs.
4) Split into train/test sets using sentence-level shuffling.
5) Fit unigram and bigram models with `<UNK>` handling.
6) Compute corpus statistics for full and train splits.
7) Evaluate cross-entropy and perplexity (unigram, bigram unsmoothed, bigram add-k).
8) Write results to `assignments/A1_foundations/outputs/`.
