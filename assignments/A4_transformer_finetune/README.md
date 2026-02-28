# Assignment 4: Transformer Fine-tuning

This assignment fine-tunes a small transformer model for sentiment classification using a public dataset (SST-2 from GLUE). It trains a lightweight model, evaluates accuracy and F1, and saves results for reporting.

## Usage

From the repository root:

```bash
python -m assignments.A4_transformer_finetune.src.run_all --help
python -m assignments.A4_transformer_finetune.src.run_all
```

## Expected Outputs

Outputs are written to `assignments/A4_transformer_finetune/outputs/`:

- `metrics.json` and `metrics.md`
- `training_args.json`
