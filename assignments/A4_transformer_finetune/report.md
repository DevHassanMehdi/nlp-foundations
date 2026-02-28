# Assignment 4 Report

## Goal

Fine-tune a pretrained transformer on a sentiment classification task and
evaluate its performance using standard metrics. The goal is to observe how a
small fine-tuning run on a limited subset of SST-2 can still achieve strong
accuracy and F1 compared to classical baselines.

## Data

- Dataset: SST-2 (GLUE)
- Size: 67,349 train / 872 validation / 1,821 test (full dataset)
- Classes: positive and negative

## Methods

- Model: DistilBERT (`distilbert-base-uncased`) for sequence classification.
- Tokenization: pretrained tokenizer with max_length=128, truncation enabled.
- Training: 1 epoch, batch_size=16, learning_rate=5e-5, 800 training samples.
- Evaluation: 200 validation samples, accuracy/precision/recall/F1.

## Results

| Metric | Value |
| --- | --- |
| Accuracy | 0.8400 |
| F1 | 0.8416 |

## Discussion

Even with a small subset of SST-2, the fine-tuned DistilBERT model reached
0.84 accuracy and 0.84 F1 after a single epoch, indicating strong transfer from
pretraining. The short runtime makes this setup practical for quick iterations,
but the results are limited by the small sample size and a single training
epoch. A full-data run with more epochs and hyperparameter tuning would likely
improve performance and stability.

## References

- Devlin et al. (BERT), 2018
- HuggingFace Transformers documentation
