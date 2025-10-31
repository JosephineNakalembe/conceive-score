from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import torch

# Load model and scaler
clf = joblib.load("clf.pkl")
scaler = joblib.load("scaler.pkl")

# Load embeddings from disk
embedding_files = {
    "rgcn": "embeddings_rgcn.pt",
    "gat": "embeddings_gat.pt",
    "tgn": "embeddings_tgn.pt",
    "vgae": "embeddings_vgae.pt",
    "gin": "embeddings_gin.pt"
}
embeddings_dict = {k: torch.load(v) for k, v in embedding_files.items()}

# Define input schema with 17 mechanistic features + patient index
class FeaturesInput(BaseModel):
    AGE: float
    BMI: float
    E2: float
    Progesterone: float
    LH: float
    FSH: float
    any_disease: float
    Weight_kg: float
    Height_m: float
    Workout_Type: float
    diet_type: float
    CycleNumber: float
    LengthofCycle: float
    EstimatedDayofOvulation: float
    LengthofLutealPhase: float
    TotalDaysofFertility: float
    Gravida: float
    patient_index: int  # used to fetch embeddings

# Initialize FastAPI app
app = FastAPI()

# -------------------- ADD CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------------------

@app.get("/")
def read_root():
    return {"message": "Conception prediction API is live 🎉"}

@app.post("/predict")
def predict(input: FeaturesInput):
    try:
        # Extract mechanistic features in order
        features = [
            input.AGE, input.BMI, input.E2, input.Progesterone, input.LH, input.FSH,
            input.any_disease, input.Weight_kg, input.Height_m, input.Workout_Type,
            input.diet_type, input.CycleNumber, input.LengthofCycle,
            input.EstimatedDayofOvulation, input.LengthofLutealPhase,
            input.TotalDaysofFertility, input.Gravida
        ]

        # Load embeddings for the given patient index
        emb_parts = []
        for emb in embeddings_dict.values():
            if input.patient_index < len(emb):
                emb_parts.append(emb[input.patient_index].cpu().numpy())
            else:
                emb_parts.append(np.zeros(emb.shape[1]))  # fallback if index out of range

        # Concatenate embeddings + features
        fused = np.concatenate(emb_parts + [np.array(features)])

        # Scale and predict
        fused_scaled = scaler.transform(fused.reshape(1, -1))
        prob = clf.predict_proba(fused_scaled)[0, 1]
        pred = int(prob >= 0.5)

        return {"probability": round(prob, 4), "prediction": pred}
    except Exception as e:
        return {"error": str(e)}
