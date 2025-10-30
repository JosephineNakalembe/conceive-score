from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI(title="Conception Probability Model API")

# Load model and scaler from pickle
try:
    with open("logistic_regression_model.pkl", "rb") as f:
        data = pickle.load(f)  # ✅ Correct loader for pickle-saved model
        model = data["model"]
        scaler = data["scaler"]
        results = data.get("results", None)  # Optional: evaluation metrics
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# Define input schema for 9 clinical features
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
        "status": "✅ Conception Probability Model API is running",
        "available_routes": ["/predict", "/metrics"]
    }

# Prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
    try:
        # Convert input to NumPy array
        features = np.array([[input_data.Age,
                              input_data.any_disease,
                              input_data.Weight,
                              input_data.Height,
                              input_data.Workout_Type,
                              input_data.diet_type,
                              input_data.CycleNumber,
                              input_data.TotalDaysofFertility,
                              input_data.Gravida]])
        
        # Apply scaler
        features_scaled = scaler.transform(features)
        
        # Predict probability of conception
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
