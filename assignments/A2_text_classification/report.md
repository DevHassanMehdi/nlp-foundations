# Assignment 2 Report

## Goal

Build a baseline sentiment classifier for movie reviews using a transparent
bag-of-words pipeline. The goal is to practice data loading, preprocessing,
feature extraction, model training, and evaluation with standard metrics.

## Data

- Dataset: `movie_reviews` (NLTK)
- Size: 2,000 documents
- Classes: positive (1,000) and negative (1,000)

## Methods

- Preprocessing: lowercase, tokenize, remove punctuation-only tokens.
- Features: bag-of-words counts; vocabulary built from training set with
  `min_count=2` and `max_vocab=10000`, plus `<UNK>` for unseen words.
- Models: Multinomial Naive Bayes (alpha=1.0) and Logistic Regression
  (lr=0.1, epochs=10, l2=0.0).
- Evaluation: 80/20 train/test split with seed=42; accuracy, precision, recall, F1.

## Results

| Model | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| Naive Bayes | 0.8125 | 0.8250 | 0.8049 | 0.8148 |
| Logistic Regression | 0.4875 | 0.0000 | 0.0000 | 0.0000 |

## Discussion

The Naive Bayes baseline performed well for a simple bag-of-words model, which
matches expectations for sentiment on this dataset. Logistic Regression failed
to learn a useful decision boundary in this configuration and effectively
predicted only the negative class, resulting in zero precision/recall for the
positive class. This suggests the optimization setup (learning rate, epochs,
and lack of regularization tuning) is insufficient; it likely needs more epochs,
feature scaling, or a different learning rate schedule.

Overall, I learned that strong baselines can come from simple generative models,
and that discriminative models are sensitive to optimization choices even with
the same features. This makes evaluation and careful hyperparameter tuning a
critical part of the workflow.

## References

- Jurafsky, D., & Martin, J. H. (Speech and Language Processing)
