from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .classic_model import train_or_load
from .transformer_model import load_transformer, predict as transformer_predict

app = FastAPI(title="Sentiment Live")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    text: str
    classic: dict
    transformer: dict


classic_model = None
transformer_model = None


@app.on_event("startup")
def _load_models() -> None:
    global classic_model, transformer_model
    classic_model = train_or_load()
    transformer_model = load_transformer()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    text = req.text.strip()
    if not text:
        return AnalyzeResponse(text="", classic={"label": "", "score": 0.0}, transformer={"label": "", "score": 0.0})

    classic_label, classic_score = classic_model.predict(text)
    trans_label, trans_score = transformer_predict(transformer_model, text)

    return AnalyzeResponse(
        text=text,
        classic={"label": classic_label, "score": round(classic_score, 4)},
        transformer={"label": trans_label, "score": round(trans_score, 4)},
    )
