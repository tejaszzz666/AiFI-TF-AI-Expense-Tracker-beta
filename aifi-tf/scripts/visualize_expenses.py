# scripts/visualize_expenses.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parents[1]
CSV_FILE = BASE / "data" / "transactions_with_category.csv"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load historical CSV
df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")

# ----------------------------
# Historical Analysis (Existing)
# ----------------------------
# Total amount per category
category_totals = df.groupby("Predicted_Category")["Amount"].sum()
print("💰 Total Amount per Category:")
print(category_totals)
print("-"*50)

# Total amount per month
df["Date"] = pd.to_datetime(df["Date"])
month_totals = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()
print("📅 Total Amount per Month:")
print(month_totals)
print("-"*50)

# Top 10 expenses
top_expenses = df.nsmallest(10, "Amount")
print("🔝 Top 10 Expenses:")
print(top_expenses[["Date","Description","Amount","Predicted_Category"]])
print("-"*50)

# Overspending alert
threshold = -1000
overspend = category_totals[category_totals < threshold]
print("⚠️ Overspending Alert! Categories exceeding threshold:")
print(overspend)
print("-"*50)

# --- Seaborn Barplot ---
plt.figure(figsize=(10,6))
sns.barplot(x=category_totals.index, y=category_totals.values, palette="Set2")
plt.title("Total Amount per Category")
plt.ylabel("Amount")
plt.xlabel("Category")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_totals.png")
plt.close()

# --- Seaborn Month Plot ---
plt.figure(figsize=(10,6))
month_totals.plot(kind="bar", color="skyblue")
plt.title("Total Amount per Month")
plt.ylabel("Amount")
plt.xlabel("Month")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "month_totals.png")
plt.close()

# --- Plotly Interactive Top Expenses ---
fig = px.bar(top_expenses, x="Description", y="Amount", color="Predicted_Category",
             title="Top 10 Expenses", text="Amount")
fig.write_html(PLOTS_DIR / "top_expenses.html")

# ----------------------------
# Next-Month Predicted Expenses
# ----------------------------
MODEL_PATH = BASE / "models" / "expense_model.pkl"
VEC_PATH = BASE / "models" / "vectorizer.pkl"

# Load model + vectorizer
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

# Example planned transactions for next month
next_month_samples = [
    "Starbucks Coffee", "Amazon Shopping", "Netflix Subscription",
    "Gym Membership", "Bus Ticket", "Electricity Bill"
]
amounts = [-250, -500, -500, -1500, -50, -1600]  # example amounts

# Vectorize and predict
X = vectorizer.transform(next_month_samples)
predicted_categories = model.predict(X)

# Create dataframe
df_pred = pd.DataFrame({
    "Description": next_month_samples,
    "Amount": amounts,
    "Predicted_Category": predicted_categories
})

# Group by category
pred_totals = df_pred.groupby("Predicted_Category")["Amount"].sum()

# Plotly interactive bar chart for predicted expenses
fig_pred = px.bar(df_pred, x="Predicted_Category", y="Amount",
                  color="Predicted_Category", text="Amount",
                  title="Next-Month Predicted Expenses")
fig_pred.write_html(PLOTS_DIR / "next_month_predicted.html")

print(f"✅ All plots saved in: {PLOTS_DIR}")
print("💡 Preview next-month predictions:")
print(df_pred)
