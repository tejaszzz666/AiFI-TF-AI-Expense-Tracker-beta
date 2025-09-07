# scripts/analyze_expenses.py
import pandas as pd
from pathlib import Path

# Paths
BASE = Path(__file__).resolve().parents[1]
CSV_PATH = BASE / "data" / "transactions_with_category.csv"

# Load data
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# Ensure Amount is numeric
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

# Add Month column
df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')

# 1️⃣ Total per category
category_summary = df.groupby('Predicted_Category')['Amount'].sum().sort_values(ascending=False)
print("💰 Total Amount per Category:")
print(category_summary)
print("\n" + "-"*50)

# 2️⃣ Total per month
monthly_summary = df.groupby('Month')['Amount'].sum()
print("📅 Total Amount per Month:")
print(monthly_summary)
print("\n" + "-"*50)

# 3️⃣ Optional: Top 5 biggest expenses
top_expenses = df[df['Amount'] < 0].sort_values('Amount').head(5)
print("🔝 Top 5 Expenses:")
print(top_expenses[['Date', 'Description', 'Amount', 'Predicted_Category']])

