# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from utils.predict_utils import predict_category

app = FastAPI(title="AI Expense Tracker API")

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "AI Expense Tracker API is running!"}

# Input model for prediction
class Transaction(BaseModel):
    description: str

# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    category = predict_category(transaction.description)
    return {
        "description": transaction.description,
        "predicted_category": category
    }
