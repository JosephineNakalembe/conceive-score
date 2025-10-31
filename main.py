from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
clf = joblib.load("clf.pkl")
scaler = joblib.load("scaler.pkl")

# Define input schema
class InputVector(BaseModel):
    embedding: list  # Expecting a list of floats (length = 160 + 17)

# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Conception prediction API is live 🎉"}

@app.post("/predict")
def predict(input: InputVector):
    try:
        vec = np.array(input.embedding).reshape(1, -1)
        vec_scaled = scaler.transform(vec)
        prob = clf.predict_proba(vec_scaled)[0, 1]
        pred = int(prob >= 0.5)
        return {"probability": round(prob, 4), "prediction": pred}
    except Exception as e:
        return {"error": str(e)}
