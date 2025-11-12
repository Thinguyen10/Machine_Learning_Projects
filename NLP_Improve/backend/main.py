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
    label: Optional[int]  # 0=negative, 1=neutral, 2=positive
    sentiment: Optional[str]  # "negative", "neutral", or "positive"
    probability: Optional[float]
    backend: str
    text: str
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
    """Report which saved artifacts and trained models are present on disk."""
    artifacts_info = {
        'vectorizer': os.path.exists('vect.joblib') or os.path.exists('backend/vect.joblib'),
        'sklearn_model': os.path.exists('model_sklearn.joblib') or os.path.exists('backend/model_sklearn.joblib') or os.path.exists('model.joblib') or os.path.exists('backend/model.joblib'),
        'keras_model': os.path.exists('model_keras.keras') or os.path.exists('backend/model_keras.keras') or os.path.exists('model_keras') or os.path.exists('backend/model_keras') or os.path.exists('keras_model') or os.path.exists('backend/keras_model'),
    }
    
    # Load metrics if available
    metrics = {}
    for model_type in ['sklearn', 'keras']:
        metrics_file = f'metrics_{model_type}.json'
        if os.path.exists(metrics_file):
            try:
                import json
                with open(metrics_file) as f:
                    metrics[model_type] = json.load(f)
            except:
                pass
        elif os.path.exists(f'backend/{metrics_file}'):
            try:
                import json
                with open(f'backend/{metrics_file}') as f:
                    metrics[model_type] = json.load(f)
            except:
                pass
    
    # Get vocab size and training info from vectorizer
    vocab_size = None
    training_samples = None
    try:
        if mw.vect is not None:
            vocab_size = len(mw.vect.vocabulary_)
            # Try to infer training samples from metrics
            if 'sklearn' in metrics and 'report' in metrics['sklearn']:
                training_samples = metrics['sklearn']['report'].get('weighted avg', {}).get('support')
    except:
        pass
    
    return {
        'artifacts': artifacts_info,
        'metrics': metrics,
        'available_models': [k for k, v in artifacts_info.items() if k.endswith('_model') and v],
        'vocab_size': vocab_size,
        'training_samples': int(training_samples) if training_samples else None
    }


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
    """Predict sentiment using pre-trained models.
    
    Args:
        text: Input text to classify
        backend: "sklearn" (fast, accurate) or "keras" (neural network)
    
    Returns:
        label: 0 (negative), 1 (neutral), or 2 (positive)
        sentiment: "negative", "neutral", or "positive"
        probability: Confidence score [0, 1]
        backend: Which model was used
        text: Original input text
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must be a non-empty string")
    
    backend_choice = req.backend or "sklearn"
    if backend_choice not in ["sklearn", "keras"]:
        raise HTTPException(status_code=400, detail="backend must be 'sklearn' or 'keras'")
    
    try:
        label, prob = mw.predict(req.text, backend=backend_choice)
        
        if label is None:
            raise HTTPException(status_code=500, detail="Model returned invalid prediction")
        
        # Map label to sentiment name
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map.get(label, "unknown")
            
        return PredictResponse(
            label=label, 
            sentiment=sentiment,
            probability=prob, 
            backend=backend_choice,
            text=req.text
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"{str(e)} Please run 'python -m backend.train_models' to train models first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
