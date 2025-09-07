# scripts/predict_expense.py
import os
import pandas as pd
import joblib
from pathlib import Path

# --- Paths ---
BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE / "models" / "expense_model.pkl"
VEC_PATH = BASE / "models" / "vectorizer.pkl"
CSV_IN = BASE / "data" / "transactions.csv"
CSV_OUT = BASE / "data" / "transactions_with_category.csv"

# --- Helper to find description column ---
def find_description_column(df):
    for c in df.columns:
        if c.strip().lower() in ("description", "desc", "narration", "details", "particulars"):
            return c
    # fallback: second column (if exists) or first
    return df.columns[1] if df.shape[1] > 1 else df.columns[0]

# --- Load model ---
if not MODEL_PATH.exists():
    raise SystemExit(f"Model not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH.name)

# --- Load vectorizer ---
vectorizer = None
if VEC_PATH.exists():
    vectorizer = joblib.load(VEC_PATH)
    print("✅ Loaded vectorizer:", VEC_PATH.name)
else:
    print("⚠️ Vectorizer not found. Model must be a pipeline to work without it.")

# --- Load transactions ---
if CSV_IN.exists():
    df = pd.read_csv(CSV_IN, encoding="utf-8-sig")
    desc_col = find_description_column(df)
    print("Using description column:", desc_col)
    texts = df[desc_col].astype(str).tolist()
else:
    print("⚠️ transactions.csv not found. Using sample transactions.")
    texts = ["Starbucks Coffee", "Electricity Bill", "Uber Ride", "Salary", "Amazon Shopping"]
    df = pd.DataFrame({"Description": texts})

# --- Predict ---
try:
    if vectorizer is not None:
        X = vectorizer.transform(texts)  # convert to sparse matrix
        preds = model.predict(X)
    else:
        preds = model.predict(texts)     # pipeline model
except Exception as e:
    raise SystemExit(f"Prediction failed. Check vectorizer/model.\nError: {e}")

# --- Save predictions ---
df["Predicted_Category"] = preds
df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print(f"✅ Predictions saved -> {CSV_OUT}")

# --- Preview ---
print("\nPreview of predictions:")
print(df.head(10).to_string(index=False))
