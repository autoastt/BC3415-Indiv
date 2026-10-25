"""Web demo for the review sentiment analyzer.

Run:   .venv/bin/uvicorn app:app --port 8000
Open:  http://localhost:8000
"""
import io
from collections import Counter
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from review_analyzer import LABELS, ReviewAnalyzer

STATIC_DIR = Path(__file__).parent / "static"
MAX_BATCH_ROWS = 2000

app = FastAPI(title="Review Sentiment Analyzer")
analyzer = ReviewAnalyzer()  # loaded once at start-up


class ReviewIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    model: str = "bert"


@app.post("/api/analyze")
def analyze(review: ReviewIn):
    if review.model not in ReviewAnalyzer.MODELS:
        raise HTTPException(400, f"model must be one of {ReviewAnalyzer.MODELS}")
    return analyzer.analyze(review.text, review.model)


@app.post("/api/batch")
async def batch(file: UploadFile = File(...), text_col: str = Form(None), model: str = Form("bert")):
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Could not read the file as CSV.")
    # Use the requested column, otherwise guess: the text column with the longest average length
    if not text_col:
        text_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        if not text_cols:
            raise HTTPException(400, "No text column found in the CSV.")
        text_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
    if text_col not in df.columns:
        raise HTTPException(400, f"Column '{text_col}' not found. Columns: {list(df.columns)}")
    df = df.head(MAX_BATCH_ROWS)

    rows = analyzer.analyze_batch(df[text_col].fillna("").astype(str), model)
    extra = df.drop(columns=text_col).astype(object)
    extra = extra.where(extra.notna(), None)  # NaN is not valid JSON
    # Aggregate per-aspect sentiment counts for the pain-point chart
    aspect_counts = {}
    for r in rows:
        for aspect, s in r["aspects"].items():
            aspect_counts.setdefault(aspect, Counter())[s] += 1
    return {
        "text_col": text_col,
        "columns": list(df.columns),
        "n": len(rows),
        "sentiment_counts": {l: sum(r["sentiment"] == l for r in rows) for l in LABELS},
        "aspects": {a: {l: c[l] for l in LABELS} for a, c in aspect_counts.items()},
        "risk_counts": dict(Counter(f for r in rows for f in r["risk_flags"])),
        "rows": [{**orig, **r} for orig, r in zip(extra.to_dict("records"), rows)],
    }


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
