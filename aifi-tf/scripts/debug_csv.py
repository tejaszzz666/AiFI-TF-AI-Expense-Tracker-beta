# scripts/debug_csv.py
import os, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, "data", "transactions.csv")

print("CSV Path:", CSV_PATH)
print("Exists:", os.path.exists(CSV_PATH))

if not os.path.exists(CSV_PATH):
    raise SystemExit("transactions.csv not found. Put it into data/ and re-run this script.")

# Try reading with utf-8-sig and auto-detect separator
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", sep=None, engine="python")
except Exception as e:
    print("Read error:", e)
    raise

print("\nFirst 8 rows:")
print(df.head(8).to_string(index=False))

print("\nColumns found:")
print(df.columns.tolist())

# show column names stripped (detect similar names)
col_candidates = {c: c.strip().lower() for c in df.columns}
print("\nStripped/lower columns:")
print(col_candidates)
