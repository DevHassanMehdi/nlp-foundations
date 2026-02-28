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

## Run Assignment 3

```bash
python -m nltk.downloader punkt punkt_tab gutenberg
python -m assignments.A3_embeddings.src.run_all --help
python -m assignments.A3_embeddings.src.run_all
```

Outputs are written to `assignments/A3_embeddings/outputs/`.

## Run Assignment 4

```bash
python -m assignments.A4_transformer_finetune.src.run_all --help
python -m assignments.A4_transformer_finetune.src.run_all
```

Outputs are written to `assignments/A4_transformer_finetune/outputs/`.

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

## Assignment 2 Code Overview

Key modules for Assignment 2 live under `assignments/A2_text_classification/src/`:

- `load_data.py`: Loads NLTK `movie_reviews` as text/label pairs.
- `preprocess.py`: Tokenization, punctuation filtering, vocabulary building, and count vectorization.
- `models.py`: Multinomial Naive Bayes and Logistic Regression from scratch.
- `evaluate.py`: Train/test split and classification metrics (accuracy, precision, recall, F1).
- `utils.py`: Reproducibility and IO helpers (seed, JSON/Markdown saves).
- `run_all.py`: Orchestrates the end-to-end classification experiment and writes outputs.

## Assignment 2 Pipeline Summary

1) Load movie reviews from NLTK.
2) Split into train/test sets.
3) Tokenize and build a vocabulary from training data.
4) Vectorize documents as bag-of-words counts.
5) Train Naive Bayes and Logistic Regression models.
6) Evaluate metrics and write results to `assignments/A2_text_classification/outputs/`.

## Assignment 3 Code Overview

Key modules for Assignment 3 live under `assignments/A3_embeddings/src/`:

- `load_data.py`: Loads Gutenberg text for training embeddings.
- `preprocess.py`: Tokenization, punctuation filtering, and sentence splitting.
- `train_embeddings.py`: Trains a Word2Vec model with configurable hyperparameters.
- `evaluate.py`: Computes word-pair similarities and nearest neighbors.
- `visualize.py`: PCA-based 2D projection for visualization.
- `utils.py`: Reproducibility and IO helpers (seed, JSON/Markdown saves).
- `run_all.py`: Runs the pipeline and saves outputs to disk.

## Assignment 3 Pipeline Summary

1) Load raw text from Gutenberg.
2) Tokenize and split into sentences.
3) Train Word2Vec embeddings.
4) Compute similarities and nearest neighbors.
5) Export PCA coordinates for visualization.
6) Write outputs to `assignments/A3_embeddings/outputs/`.

## Assignment 4 Code Overview

Key modules for Assignment 4 live under `assignments/A4_transformer_finetune/src/`:

- `run_all.py`: Loads SST-2, tokenizes, fine-tunes a transformer, and evaluates metrics.
- `utils.py`: Minimal IO helpers for outputs.

## Assignment 4 Pipeline Summary

1) Load SST-2 dataset from GLUE via HuggingFace Datasets.
2) Tokenize sentences with a pretrained transformer tokenizer.
3) Fine-tune a sequence classification model (DistilBERT by default).
4) Evaluate accuracy, precision, recall, and F1.
5) Write outputs to `assignments/A4_transformer_finetune/outputs/`.
