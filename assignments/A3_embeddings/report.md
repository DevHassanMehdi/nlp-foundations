# Assignment 3 Report

## Goal

Train and analyze word embeddings to see how distributional similarity captures
semantic and contextual relationships in text. The objective is to build a small
Word2Vec model and interpret similarity scores, nearest neighbors, and a 2D
projection to understand what the model learns from a single novel.

## Data

- Dataset: `austen-emma.txt` (NLTK Gutenberg)
- Size: 161,521 tokens, 7,897 vocabulary

## Methods

- Preprocessing: lowercase, tokenize, remove punctuation-only tokens; split into sentences.
- Embedding model (Word2Vec): skip-gram (`sg=1`), vector_size=100, window=5,
  min_count=2, epochs=5, seed=42.
- Evaluation (similarities, neighbors): cosine similarities for selected word
  pairs and top-5 nearest neighbors for key terms.
- Visualization (PCA): SVD-based PCA projection of the top-50 most frequent
  vocabulary items.

## Results

- Similarity highlights:
  - `mr`–`mrs`: 0.9843
  - `man`–`woman`: 0.9336
  - `emma`–`harriet`: 0.8659
  - `good`–`bad`: 0.8703
- Nearest neighbor examples:
  - `emma`: then, harriet, he, she, isabella
  - `good`: pretty, fine, serious, charming, bad
  - `friend`: opinion, account, situation, mother, sister
- PCA observations:
  - The most frequent words cluster near each other, suggesting the model
    captures broad contextual similarity among common narrative terms.

## Discussion

The embeddings capture strong associations for frequent and contextually related
terms (e.g., `mr`–`mrs`, `man`–`woman`, `emma`–`harriet`). Some pairs like
`good`–`bad` are also highly similar, reflecting that antonyms often share
contexts in natural language. Nearest neighbors show that high-frequency words
can dominate local neighborhoods, which is a limitation when training on a
single novel. Still, the model learns meaningful relationships without labels,
demonstrating how distributional context can approximate semantics.

## References

- Jurafsky, D., & Martin, J. H. (Speech and Language Processing)
