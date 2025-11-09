"""
FastAPI Backend for LSTM Text Generation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uvicorn
import os

from train_predict import load_training_artifacts, generate_sentence

app = FastAPI(
    title="LSTM Text Generation API",
    description="Generate text using LSTM next-word prediction model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lstm_model = None
lstm_tokenizer = None
lstm_config = None
LSTM_MODEL_DIR = "trained_model"


class TextGenerationRequest(BaseModel):
    seed_text: str = Field(..., min_length=1, max_length=500)
    num_words: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


class TextGenerationResponse(BaseModel):
    seed_text: str
    generated_text: str
    num_words_generated: int


class LSTMStatusResponse(BaseModel):
    model_loaded: bool
    window_size: Optional[int] = None
    vocab_size: Optional[int] = None
    model_info: Optional[Dict] = None


@app.on_event("startup")
async def startup_event():
    global lstm_model, lstm_tokenizer, lstm_config
    model_path = os.path.join(LSTM_MODEL_DIR, "lstm_model.h5")
    
    if os.path.exists(LSTM_MODEL_DIR) and os.path.exists(model_path):
        try:
            lstm_model, lstm_tokenizer, lstm_config = load_training_artifacts(LSTM_MODEL_DIR)
            print(f"✓ LSTM model loaded from {LSTM_MODEL_DIR}")
        except Exception as e:
            print(f"✗ Failed to load LSTM: {e}")
    else:
        print(f"⚠ Run 'python example_pipeline.py' to train the model")


@app.get("/")
async def root():
    return {"message": "LSTM Text Generation API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "lstm_model_loaded": lstm_model is not None}


@app.get("/lstm/status", response_model=LSTMStatusResponse)
async def get_lstm_status():
    if lstm_model is None:
        return {
            "model_loaded": False,
            "window_size": None,
            "vocab_size": None,
            "model_info": {"message": "LSTM model not loaded"}
        }
    return {
        "model_loaded": True,
        "window_size": lstm_config.get("window_size"),
        "vocab_size": lstm_config.get("vocab_size"),
        "model_info": lstm_config
    }


@app.post("/lstm/generate", response_model=TextGenerationResponse)
async def generate_text_endpoint(request: TextGenerationRequest):
    if lstm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        generated = generate_sentence(
            model=lstm_model,
            seed_text=request.seed_text,
            tokenizer=lstm_tokenizer,
            window_size=lstm_config["window_size"],
            num_words=request.num_words,
            temperature=request.temperature
        )
        
        words_generated = len(generated.split()) - len(request.seed_text.split())
        
        return {
            "seed_text": request.seed_text,
            "generated_text": generated,
            "num_words_generated": max(0, words_generated)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lstm/examples")
async def get_lstm_examples():
    return {
        "simple": ["the quick brown", "once upon a", "it was a"],
        "questions": ["what is the", "how can we", "where did you"],
        "actions": ["he walked to", "she began to", "they decided to"],
        "descriptions": ["the beautiful garden", "a dark and", "an old book"]
    }
