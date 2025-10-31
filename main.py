# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# ==========================================
# Load trained model and scaler
# ==========================================
with open("clf.pkl", "rb") as f:
    clf = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================================
# Define expected input
# ==========================================
class PatientFeatures(BaseModel):
    AGE: float
    BMI: float
    E2: float
    Progesterone: float
    LH: float
    FSH: float
    any_disease: int  # 0/1
    Weight_kg: float
    Height_m: float
    Workout_Type: int  # map categories to int
    diet_type: int     # map categories to int
    CycleNumber: int
    LengthofCycle: float
    EstimatedDayofOvulation: float
    LengthofLutealPhase: float
    TotalDaysofFertility: float
    Gravida: int

app = FastAPI(title="Conceive Score API")

# ==========================================
# Predict endpoint
# ==========================================
@app.post("/predict")
def predict(data: PatientFeatures):
    # Convert input to array in correct order
    X = np.array([[
        data.AGE,
        data.BMI,
        data.E2,
        data.Progesterone,
        data.LH,
        data.FSH,
        data.any_disease,
        data.Weight_kg,
        data.Height_m,
        data.Workout_Type,
        data.diet_type,
        data.CycleNumber,
        data.LengthofCycle,
        data.EstimatedDayofOvulation,
        data.LengthofLutealPhase,
        data.TotalDaysofFertility,
        data.Gravida
    ]])

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict probability
    prob = clf.predict_proba(X_scaled)[:, 1][0]

    return {"probability_of_conception": float(prob)}
