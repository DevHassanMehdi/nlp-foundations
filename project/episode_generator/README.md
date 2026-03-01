# Episode Generator (Three Models, Same Prompt)

This project compares three local generation models side-by-side using the same prompt and the same episode schema.

- Model A: N-gram language model (word-level)
- Model B: Word-level LSTM (planner-driven, constrained decoding)
- Model C: Transformer LM (Qwen2.5 Instruct, locally hosted with fallback tiers)

All outputs use the same episode schema:
- TITLE
- EPISODE ITEMS
- LOGLINE
- PLOT OUTLINE
- SCENE BREAKDOWN
- SCRIPT SAMPLE

## Setup

```bash
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab gutenberg
```

## Data Prep

```bash
python -m project.episode_generator.scripts.data_prep
```

## Train Models

```bash
python -m project.episode_generator.scripts.train_ngram
```

High-quality LSTM training (targeted for i7-9700F + RTX 2070):

```bash
python -m project.episode_generator.scripts.train_word_lstm \
  --epochs 12 \
  --seq_len 96 \
  --batch_size 32 \
  --grad_accum_steps 2 \
  --max_tokens 260000 \
  --max_vocab 18000 \
  --emb_dim 224 \
  --hidden_size 448 \
  --num_layers 2 \
  --dropout 0.28 \
  --lr 7e-4 \
  --weight_decay 1e-4 \
  --device cuda \
  --seed 42
```

If you get CUDA OOM, reduce `--batch_size` to `24` or `16`.

## Generate (CLI)

```bash
python -m project.episode_generator.scripts.generate \
  --prompt "On a nice, peaceful morning, Janene goes for a walk and has a strange encounter with a bear in the woods." \
  --temperature 0.72 \
  --seed 42
```

## Live Demo (FastAPI + Frontend)

Backend:

```bash
python -m uvicorn project.episode_generator.backend.app.main:app --reload
```

Frontend:

```bash
cd project/episode_generator/frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

Note: On first backend startup, the transformer model is downloaded once from Hugging Face and cached locally.
Default is `Qwen/Qwen2.5-1.5B-Instruct`, with automatic fallback to `Qwen/Qwen2.5-0.5B-Instruct` and then `distilgpt2` if memory is limited.

In my experiments, this transformer pipeline was the strongest performer for fluency and section-level coherence.
