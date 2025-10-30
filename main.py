from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI(title="Conception Probability Model API")

# Load model and scaler from pickle
with open("late_fusion_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]

# Define input schema
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
    return {"status": "Conception Probability Model API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
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
