# Sentiment Live Demo

A local, side-by-side sentiment analysis demo that compares a classic TF-IDF + Logistic Regression model with a transformer (DistilBERT). The backend is FastAPI; the frontend is a minimal modern web UI.

## Backend (FastAPI)

```bash
python -m pip install -r requirements.txt
python -m uvicorn project.sentiment_live.backend.app.main:app --reload
```

The backend runs at `http://localhost:8000`.

## Frontend (Node/Vite)

```bash
cd project/sentiment_live/frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`.

## Notes

- The first backend run downloads the SST-2 dataset and the transformer weights.
- The classic model trains on a small SST-2 subset for fast startup and is cached in `project/sentiment_live/backend/app/model_cache/`.
