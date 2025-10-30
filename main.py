from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI(title="Late Fusion Conception Model API")

# Load model and scaler from pickle
try:
    with open("late_fusion_model.pkl", "rb") as f:
        data = pickle.load(f)
        model = data["model"]
        scaler = data["scaler"]
        results = data.get("results", None)
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Define input schema for 9 known features
class ModelInput(BaseModel):
    Age: float
    any_disease: int
    Weight: float
    Height: float
    Workout_Type: int
    diet_type: int
    CycleNumber: int
    TotalDaysofFertility: int
    Gravida: int

# Root endpoint
@app.get("/")
def read_root():
    return {
        "status": "✅ Late Fusion Model API is running",
        "available_routes": ["/predict", "/metrics"]
    }

# Prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # Step 1: Create a zero-filled 160-dimensional vector
        full_input = np.zeros(160)

        # Step 2: Insert known features into positions 0–8
        full_input[0] = input_data.Age
        full_input[1] = input_data.any_disease
        full_input[2] = input_data.Weight
        full_input[3] = input_data.Height
        full_input[4] = input_data.Workout_Type
        full_input[5] = input_data.diet_type
        full_input[6] = input_data.CycleNumber
        full_input[7] = input_data.TotalDaysofFertility
        full_input[8] = input_data.Gravida

        # Step 3: Scale and predict
        features_scaled = scaler.transform([full_input])
        prob = model.predict_proba(features_scaled)[0][1]
        prediction = int(prob >= 0.5)

        return {
            "probability": round(prob, 4),
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# Optional: expose evaluation metrics
@app.get("/metrics")
def get_metrics():
    if results is not None:
        return results.to_dict()
    else:
        raise HTTPException(status_code=404, detail="Evaluation metrics not found")
