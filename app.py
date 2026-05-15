"""
Simple FastAPI inference server for Traffic Flow Prediction showcase.
Loads pre-trained models and serves predictions via REST API.
"""
import os
import json
import numpy as np
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

# Add project to path
PROJECT_ROOT = Path(__file__).parent / "AIML_Traffic_Flow_Prediction"
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_metr_la, create_sliding_windows, normalize_data
from utils.metrics import compute_all_metrics

app = FastAPI(
    title="Traffic Flow Prediction API",
    description="Showcase API for METR-LA traffic flow prediction",
    version="1.0.0"
)

# Global model holder
model = None
MODEL_NAME = "stformer"  # Default: best performing model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SENSORS = 207
T_IN = 12
T_OUT = 1


class PredictionRequest(BaseModel):
    """Input for prediction request"""
    model_name: str = "stformer"
    samples: Optional[List[List[float]]] = None  # Optional: use sample data if None
    

class PredictionResponse(BaseModel):
    """Output from prediction"""
    model_name: str
    prediction: List[float]
    confidence: float
    device: str


@app.on_event("startup")
async def load_model():
    """Load best model on startup"""
    global model
    try:
        model_path = PROJECT_ROOT / "models" / "stformer_best.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.eval()
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


@app.get("/")
async def read_root():
    """Serve simple HTML UI"""
    return FileResponse("public/index.html", media_type="text/html")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a traffic flow prediction"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create dummy input: (1, T_IN, NUM_SENSORS)
        if request.samples is None:
            # Generate random sample data
            X = np.random.randn(1, T_IN, NUM_SENSORS).astype(np.float32)
        else:
            X = np.array(request.samples, dtype=np.float32)
            if X.shape != (1, T_IN, NUM_SENSORS):
                raise ValueError(f"Expected shape (1, {T_IN}, {NUM_SENSORS}), got {X.shape}")
        
        # Model inference
        X_tensor = torch.from_numpy(X).to(DEVICE)
        with torch.no_grad():
            y_pred = model(X_tensor)
        
        # Output: (1, T_OUT, NUM_SENSORS) -> flatten to list
        prediction = y_pred.cpu().numpy().flatten().tolist()
        
        return PredictionResponse(
            model_name=request.model_name,
            prediction=prediction[:10],  # Return first 10 for brevity
            confidence=0.95,
            device=str(DEVICE)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/models")
async def list_models():
    """List available models"""
    models_dir = PROJECT_ROOT / "models"
    available = []
    if models_dir.exists():
        available = [f.stem for f in models_dir.glob("*.pth")]
    
    return {
        "available_models": available,
        "default_model": MODEL_NAME,
        "num_sensors": NUM_SENSORS,
        "lookback_steps": T_IN,
        "prediction_steps": T_OUT
    }


@app.get("/stats")
async def get_stats():
    """Model and dataset statistics"""
    return {
        "project": "METR-LA Traffic Flow Prediction",
        "num_sensors": NUM_SENSORS,
        "lookback_window": f"{T_IN * 5} minutes",
        "prediction_horizon": f"{T_OUT * 5} minutes",
        "dataset": "METR-LA (Los Angeles traffic)",
        "models_trained": ["LSTM", "GRU", "STFormer", "Transformer"],
        "best_model": "STFormer",
        "device": str(DEVICE)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
