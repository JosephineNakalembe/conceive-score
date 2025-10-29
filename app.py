from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Conception Probability Model API")

# Load your trained model
with open("late_fusion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the input schema
class ModelInput(BaseModel):
    Age: float
    any_disease: int            # 0 for No, 1 for Yes
    Weight: float
    Height: float
    Workout_Type: int           # encoded as integer
    diet_type: int              # encoded as integer
    CycleNumber: int
    TotalDaysofFertility: int
    Gravida: int

# Prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
    # Convert input to the format your model expects
    features = np.array([[input_data.Age,
                          input_data.any_disease,
                          input_data.Weight,
                          input_data.Height,
                          input_data.Workout_Type,
                          input_data.diet_type,
                          input_data.CycleNumber,
                          input_data.TotalDaysofFertility,
                          input_data.Gravida]])
    
    # Make prediction
    prediction = model.predict(features)
    
    return {"prediction": prediction.tolist()}