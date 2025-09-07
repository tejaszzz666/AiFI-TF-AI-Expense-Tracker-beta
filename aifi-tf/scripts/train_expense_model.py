import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "expense_model.pkl")
VEC_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# Sample categories (later you can expand this)
CATEGORY_MAP = {
    "coffee": "Food & Drink",
    "uber": "Transport",
    "salary": "Income",
    "amazon": "Shopping",
    "bill": "Utilities",
    "pizza": "Food & Drink"
}

# Load data
df = pd.read_csv(DATA_PATH)

# Auto-generate categories
def label_category(desc):
    desc = desc.lower()
    for k, v in CATEGORY_MAP.items():
        if k in desc:
            return v
    return "Other"

df["Category"] = df["Description"].apply(label_category)

# Train TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Description"])
y = df["Category"]

model = LogisticRegression()
model.fit(X, y)

# Save
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VEC_PATH)

print("✅ Model trained and saved to", MODEL_PATH)
