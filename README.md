# TIES4200 NLP Course Repository

Author: Hassan Mehdi

This repository contains all assignments and the main project for TIES4200 (Independent NLP). Assignment 1 maps to Jurafsky & Martin Chapters 2–3 by implementing tokenization, n-gram language models, and evaluation with cross-entropy and perplexity.

## Setup

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
python -m nltk.downloader punkt gutenberg
```

## Run Assignment 1

```bash
python -m assignments.A1_foundations.src.run_all --help
python -m assignments.A1_foundations.src.run_all
```

Outputs are written to `assignments/A1_foundations/outputs/`.
