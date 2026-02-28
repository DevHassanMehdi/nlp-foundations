# Assignment 3: Word Embedding Analysis

This assignment trains and analyzes word embeddings using a small public-domain corpus. It includes similarity queries, nearest neighbors, and a PCA-based 2D projection for visualization.

## Usage

From the repository root:

```bash
python -m assignments.A3_embeddings.src.run_all --help
python -m assignments.A3_embeddings.src.run_all
```

If NLTK resources are missing, run:

```bash
python -m nltk.downloader punkt punkt_tab gutenberg
```

## Expected Outputs

Outputs are written to `assignments/A3_embeddings/outputs/`:

- `similarities.json` and `similarities.md` for word-pair similarities
- `neighbors.json` and `neighbors.md` for nearest neighbors
- `pca_coords.csv` for 2D coordinates suitable for plotting
