# %% [markdown]
# # SQL Demo Runner
# Loads the cleaned churn data into a local SQLite database
# (`data/processed/churn.db`) and executes every query in
# `02_business_analytics_queries.sql`, printing the results. No database
# server required — this works in Jupyter, Spyder, or plain `python`.
#
# Run 01_data_collection_cleaning.py (in ../python/) first so
# `churn_cleaned.csv` exists.

# %%
import os
import sqlite3
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DB_PATH = os.path.join(PROCESSED_DIR, "churn.db")

# %% [markdown]
# ## 1. Load cleaned CSV into SQLite

# %%
csv_path = os.path.join(PROCESSED_DIR, "churn_cleaned.csv")
df = pd.read_csv(csv_path)

table_cols = [
    "customer_id", "state", "account_length", "area_code", "international_plan",
    "voice_mail_plan", "number_vmail_messages", "total_day_minutes", "total_day_calls",
    "total_day_charge", "total_eve_minutes", "total_eve_calls", "total_eve_charge",
    "total_night_minutes", "total_night_calls", "total_night_charge",
    "total_intl_minutes", "total_intl_calls", "total_intl_charge",
    "customer_service_calls", "churn_flag", "total_charge", "total_minutes",
]
table_cols = [c for c in table_cols if c in df.columns]
sql_df = df[table_cols].rename(columns={"churn_flag": "churn"})

conn = sqlite3.connect(DB_PATH)
sql_df.to_sql("customers", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_state ON customers(state);")
conn.execute("CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn);")
conn.commit()
print(f"Loaded {len(sql_df)} rows into {DB_PATH} (table: customers)")

# %% [markdown]
# ## 2. Run every query in the .sql file and print results

# %%
sql_file = os.path.join(SCRIPT_DIR, "02_business_analytics_queries.sql")
with open(sql_file, "r") as f:
    sql_text = f.read()

# Strip comment lines FIRST, then split on semicolons that end a statement
# (simple split is fine here since our .sql file has no semicolons inside
# string literals)
sql_no_comments = "\n".join(
    line for line in sql_text.splitlines() if not line.strip().startswith("--")
)
statements = [s.strip() for s in sql_no_comments.split(";") if s.strip()]

for i, clean_stmt in enumerate(statements, 1):
    print("\n" + "=" * 80)
    print(f"QUERY {i}:")
    print(clean_stmt[:200] + ("..." if len(clean_stmt) > 200 else ""))
    print("-" * 80)
    try:
        result = pd.read_sql_query(clean_stmt, conn)
        print(result.head(15).to_string(index=False))
        if len(result) > 15:
            print(f"... ({len(result)} rows total)")
    except Exception as e:
        print(f"[Skipped — not a SELECT or needs adjustment: {e}]")

conn.close()
print(f"\nDone. Database saved at: {DB_PATH} (open with DB Browser for SQLite if you'd like a GUI).")
