from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, List
from fastapi.middleware.cors import CORSMiddleware

import os
# use absolute package imports to ensure uvicorn imports resolve
from backend.processing import preview_data, process, preview_text
from backend.training import train_sklearn, train_keras
from backend.model import ModelWrapper, save_vectorizer, save_model

app = FastAPI(title="NLP Sentiment API")

# Allow local frontend dev to call the API (React / Vite default ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str
    backend: Optional[str] = "sklearn"


class PredictResponse(BaseModel):
    label: Optional[int]
    probability: Optional[float]
    backend: str


class TrainRequest(BaseModel):
    csv_path: Optional[str] = "data.csv"
    backend: Optional[str] = "sklearn"
    epochs: Optional[int] = 5
    batch_size: Optional[int] = 32


class TrainResponse(BaseModel):
    backend: str
    metrics: Any
    model_path: str
    vect_path: str


mw = ModelWrapper()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/preview")
def preview(csv_path: Optional[str] = "data.csv", n: int = 3):
    """Return a small preview (original, cleaned, tokens, top_terms) sampled from the CSV.

    This mirrors the Streamlit preview used in the original `app.py`.
    """
    try:
        df, _ = preview_data(csv_path, n=n)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Convert DataFrame to JSON-serializable structure
    out = []
    for _, row in df.iterrows():
        item = {
            'original': str(row.get('original', '')),
            'cleaned': str(row.get('cleaned', '')),
            'tokens': list(row.get('tokens', [])) if row.get('tokens') is not None else [],
            'top_terms': list(row.get('top_terms', [])) if row.get('top_terms') is not None else []
        }
        # include label if present
        if 'sentiment' in row.index:
            item['sentiment'] = row.get('sentiment')
        out.append(item)

    return {'preview': out}


@app.get("/transform")
def transform(text: str):
    """Return cleaned text and tokens for a single input (mirrors Streamlit preview_text)."""
    try:
        cleaned, tokens = preview_text(text)
        return {'cleaned': cleaned, 'tokens': tokens}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/artifacts")
def artifacts():
    """Report which saved artifacts are present on disk."""
    exists = {
        'vect': True if os.path.exists('vect.joblib') else False,
        'sklearn_model': True if os.path.exists('model.joblib') else False,
        'keras_model': True if os.path.exists('keras_model') else False,
    }
    return exists


@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """Run preprocessing and training. Saves vectorizer and model artifacts to repo root.

    WARNING: training can be slow and may require TensorFlow if `backend` is `keras`.
    """
    csv_path = req.csv_path or 'data.csv'
    backend_choice = req.backend or 'sklearn'

    # Run preprocessing
    try:
        vect, Xtr, Xte, ytr, yte = process(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Preprocessing failed: {e}')

    # Train model
    try:
        if backend_choice == 'sklearn':
            metrics, trained_model = train_sklearn(Xtr, Xte, ytr, yte)
            model_path = 'model.joblib'
            save_model(trained_model, model_path, backend='sklearn')
        else:
            # keras training may require TF; training returns (metrics, model)
            metrics, trained_model = train_keras(Xtr, Xte, ytr, yte, epochs=req.epochs, batch_size=req.batch_size)
            model_path = 'keras_model'
            save_model(trained_model, model_path, backend='keras')

        vect_path = 'vect.joblib'
        save_vectorizer(vect, vect_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Training failed: {e}')

    return TrainResponse(backend=backend_choice, metrics=metrics, model_path=model_path, vect_path=vect_path)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must be a non-empty string")
    try:
        label, prob = mw.predict(req.text, backend=req.backend)
        return PredictResponse(label=label, probability=prob, backend=req.backend)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
