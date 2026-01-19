# Assignment 1 Report

## Goal

Implement a basic NLP pipeline covering tokenization, corpus statistics, and
unigram/bigram language modeling with evaluation via cross-entropy and perplexity.

## Data

- Dataset: `austen-emma.txt` (NLTK Gutenberg)
- Size (full): 161,521 tokens, 7,897 vocabulary
- Size (train): 145,380 tokens, 7,517 vocabulary

## Methods

- Tokenization: lowercasing, word-level tokenization, punctuation filtering; sentence boundaries with `. ! ?`.
- Language models: unigram and bigram counts with `<UNK>` and `<s>`.
- Smoothing (add-k): bigram add-k with k=0.1.
- Evaluation: cross-entropy and perplexity on held-out test set.

## Results

| Model   | add_k | cross_entropy | perplexity |
| ------- | ----- | ------------- | ---------- |
| Unigram | 0.0   | 8.6487        | 401.3383   |
| Bigram  | 0.0   | Infinity      | Infinity   |
| Bigram  | 0.1   | 8.1119        | 276.6395   |

## Experiments

- Default run on `austen-emma.txt` with `test_ratio=0.1` and `seed=42`.
- Rare words mapped to `<UNK>` using `min_count=2`; sentence starts use `<s>`.
- Bigram evaluated with add-k smoothing at `k=0.0` (unsmoothed) and `k=0.1`.

## Discussion

This exercise showed how sensitive n-gram models are to data sparsity. The
unsmoothed bigram produced infinite cross-entropy because even in a large text,
many bigrams are unseen at test time. Adding a small amount of smoothing
immediately fixes that and yields a lower perplexity than the unigram baseline,
which matches the intuition that local context improves predictions.

I also learned that tokenization choices matter: filtering punctuation and
deciding sentence boundaries directly affect which bigrams are counted. The
`<UNK>` mapping is a simple but important step to keep the model robust to rare
words without exploding the vocabulary. Finally, the gap between unigram and
smoothed bigram perplexity provides a concrete, quantitative way to see how
context helps and why smoothing is essential for reliable evaluation.

## References

- Jurafsky, D., & Martin, J. H. (Speech and Language Processing), Chapters 2–3
