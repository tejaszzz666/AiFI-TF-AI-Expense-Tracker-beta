# scripts/predict_next_month.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression

BASE = Path(__file__).resolve().parents[1]
CSV_FILE = BASE / "data" / "transactions_with_category.csv"

# Load data
df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
df["Date"] = pd.to_datetime(df["Date"])

# Group by month and category
df["Month"] = df["Date"].dt.to_period("M")
monthly = df.groupby(["Month", "Predicted_Category"])["Amount"].sum().reset_index()

# Prepare data for regression per category
predictions = []
for cat in monthly["Predicted_Category"].unique():
    cat_data = monthly[monthly["Predicted_Category"] == cat]
    cat_data = cat_data.sort_values("Month")
    
    # Encode months as integers
    X = np.arange(len(cat_data)).reshape(-1,1)
    y = cat_data["Amount"].values
    
    # Fit simple linear regression
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next month
    next_month_index = np.array([[len(cat_data)]])
    next_amount = model.predict(next_month_index)[0]
    
    predictions.append({
        "Category": cat,
        "Predicted_Amount_Next_Month": round(next_amount, 2)
    })

pred_df = pd.DataFrame(predictions)
print("📈 Predicted Expenses for Next Month:")
print(pred_df)

# Overspending alert
threshold = -1000
alert = pred_df[pred_df["Predicted_Amount_Next_Month"] < threshold]
if not alert.empty:
    print("\n⚠️ Warning! Categories likely to overspend next month:")
    print(alert)
else:
    print("\n✅ No overspending predicted.")
