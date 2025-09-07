# backend/utils/predict_utils.py
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "expense_model.pkl"
VEC_PATH = BASE / "models" / "vectorizer.pkl"

# Load model & vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

def predict_category(description):
    """Predict category for a single transaction description"""
    X = vectorizer.transform([description])
    pred = model.predict(X)
    return pred[0]
