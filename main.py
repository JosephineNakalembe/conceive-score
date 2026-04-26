from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import torch

clf = joblib.load("clf.pkl")
scaler = joblib.load("scaler.pkl")

embedding_files = {
    "rgcn": "embeddings_rgcn.pt",
    "gat": "embeddings_gat.pt",
    "tgn": "embeddings_tgn.pt",
    "vgae": "embeddings_vgae.pt",
    "gin": "embeddings_gin.pt"
}

embeddings_dict = {k: torch.load(v) for k, v in embedding_files.items()}

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route("/")
def home():
    return jsonify({"message": "Conception prediction API is live 🎉"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Optional: basic validation
        required_fields = [
            "AGE", "BMI", "E2", "Progesterone", "LH", "FSH",
            "any_disease", "Weight_kg", "Height_m", "Workout_Type",
            "diet_type", "CycleNumber", "LengthofCycle",
            "EstimatedDayofOvulation", "LengthofLutealPhase",
            "TotalDaysofFertility", "Gravida", "patient_index"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Extract features
        features = [
            data["AGE"], data["BMI"], data["E2"], data["Progesterone"],
            data["LH"], data["FSH"], data["any_disease"], data["Weight_kg"],
            data["Height_m"], data["Workout_Type"], data["diet_type"],
            data["CycleNumber"], data["LengthofCycle"],
            data["EstimatedDayofOvulation"], data["LengthofLutealPhase"],
            data["TotalDaysofFertility"], data["Gravida"]
        ]

        patient_index = data["patient_index"]

        emb_parts = []
        for emb in embeddings_dict.values():
            if patient_index < len(emb):
                emb_parts.append(emb[patient_index].cpu().numpy())
            else:
                emb_parts.append(np.zeros(emb.shape[1]))

        fused = np.concatenate(emb_parts + [np.array(features)])

        fused_scaled = scaler.transform(fused.reshape(1, -1))
        prob = clf.predict_proba(fused_scaled)[0, 1]
        pred = int(prob >= 0.5)

        return jsonify({
            "probability": round(float(prob), 4),
            "prediction": pred
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)