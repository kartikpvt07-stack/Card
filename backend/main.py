from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Cardiovascular Disease Prediction API")

# Setup CORS to allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded successfully.")
    else:
        print("Model and Scaler NOT FOUND. Please run the notebook to generate them.")

from pydantic import BaseModel, Field

class CardioFeatures(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: int = Field(..., ge=1, le=2)
    height: float = Field(..., ge=50, le=250)
    weight: float = Field(..., ge=10, le=300)
    ap_hi: float = Field(..., ge=40, le=250)
    ap_lo: float = Field(..., ge=20, le=200)
    cholesterol: int = Field(..., ge=1, le=3)
    gluc: int = Field(..., ge=1, le=3)
    smoke: int = Field(..., ge=0, le=1)
    alco: int = Field(..., ge=0, le=1)
    active: int = Field(..., ge=0, le=1)

@app.get("/")
def read_root():
    return {"message": "Welcome to Cardiovascular Disease Prediction API"}

@app.post("/predict")
def predict(features: CardioFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model/Scaler not loaded. Run notebook first.")
    
    # Convert input to DataFrame
    input_dict = features.dict()
    # Ensure correct order of columns as in training
    columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 
               'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    
    df = pd.DataFrame([input_dict], columns=columns)
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    # Predict
    prediction = model.predict(df_scaled)
    prob = model.predict_proba(df_scaled)[0][1] if hasattr(model, 'predict_proba') else None
    
    return {
        "prediction": int(prediction[0]),
        "probability": float(prob) if prob is not None else None,
        "message": "High risk of cardiovascular disease" if prediction[0] == 1 else "Low risk of cardiovascular disease"
    }
