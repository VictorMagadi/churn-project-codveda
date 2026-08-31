# %% [markdown]
# # Level 1 - Task 2: Data Collection and Cleaning
# Codveda Business Analytics Internship
#
# Works identically in **Jupyter Notebook**, **Spyder**, and as a plain script.
# Cell boundaries are marked with `# %%`.
#
# Objectives covered:
# - Collect data from a file source (CSV -> stand-in for "Excel/DB/API" source)
# - Handle missing values and outliers
# - Standardize and normalize data
# - Distinguish structured vs unstructured data

# %%
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

# Resolve paths relative to this script so it works no matter where you launch it from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Project root:", PROJECT_ROOT)

# %% [markdown]
# ## 1. Data Collection
# Simulating collection from multiple structured sources: here we combine the
# 80% and 20% BigML churn splits back into one raw pool, tag their origin, and
# treat this as our "data warehouse extract". In a real pipeline this cell is
# where you'd swap in `pd.read_sql(...)` for a database or `requests.get(...)`


# %%
df_80 = pd.read_csv(os.path.join(RAW_DIR, "churn-bigml-80.csv"))
df_20 = pd.read_csv(os.path.join(RAW_DIR, "churn-bigml-20.csv"))
df_80["source_split"] = "train_80"
df_20["source_split"] = "holdout_20"

df_raw = pd.concat([df_80, df_20], ignore_index=True)
print(f"Combined raw records: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
df_raw.head()

# %% [markdown]
# ## 2. Initial Profiling — structured vs unstructured check
# This dataset is fully **structured** (tabular, typed columns). We confirm
# dtypes and note there is no free-text/unstructured field here (that would
# require NLP text-cleaning instead of the numeric approach below).

# %%
print(df_raw.dtypes)
print("\nMissing values per column:")
print(df_raw.isnull().sum())
print(f"\nDuplicate rows: {df_raw.duplicated().sum()}")

# %% [markdown]
# ## 3. Clean column names (snake_case, no spaces)

# %%
def to_snake(col):
    return (
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

df = df_raw.copy()
df.columns = [to_snake(c) for c in df.columns]
print(df.columns.tolist())

# %% [markdown]
# ## 4. Handle missing values
# The BigML churn dataset is naturally complete, but production data rarely
# is — so this pipeline is written to be defensive regardless. Numeric
# columns are imputed with the median (robust to outliers); categoricals
# with the mode.

# %%
missing_before = df.isnull().sum().sum()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
# churn is boolean/target - keep separate from feature imputation logic if needed
print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} missing values with median={median_val}")

for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(mode_val)
        print(f"Filled {col} missing values with mode={mode_val}")

missing_after = df.isnull().sum().sum()
print(f"\nTotal missing values: {missing_before} -> {missing_after}")

# %% [markdown]
# ## 5. Remove exact duplicate rows

# %%
before_rows = len(df)
df = df.drop_duplicates()
print(f"Dropped {before_rows - len(df)} duplicate rows. Remaining: {len(df)}")

# Assign a stable unique customer_id now, before any further transforms —
# every downstream script joins on this instead of business columns
# (state/account_length are NOT unique keys on their own).
df = df.reset_index(drop=True)
df.insert(0, "customer_id", [f"CUST{i:05d}" for i in range(len(df))])

# %% [markdown]
# ## 6. Outlier detection & handling (IQR method)
# We flag outliers on the key usage/spend columns rather than blindly
# deleting rows (deleting real high-usage customers would bias a churn
# model). Instead we **cap (winsorize)** values at the IQR fences and keep a
# flag column so downstream models can use it as a feature if useful.

# %%
outlier_candidate_cols = [
    "total_day_minutes", "total_day_charge",
    "total_eve_minutes", "total_eve_charge",
    "total_night_minutes", "total_night_charge",
    "total_intl_minutes", "total_intl_charge",
    "customer_service_calls",
]
outlier_candidate_cols = [c for c in outlier_candidate_cols if c in df.columns]

outlier_summary = {}
for col in outlier_candidate_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_summary[col] = {"lower_fence": lower, "upper_fence": upper, "n_outliers": n_outliers}
    df[col] = df[col].clip(lower=lower, upper=upper)  # winsorize / cap

outlier_report = pd.DataFrame(outlier_summary).T
print(outlier_report)

# %% [markdown]
# ## 7. Standardize categorical values

# %%
for col in ["international_plan", "voice_mail_plan"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

if "churn" in df.columns:
    df["churn"] = df["churn"].astype(bool)
    df["churn_flag"] = df["churn"].astype(int)  # numeric version for modeling/BI

df["state"] = df["state"].astype(str).str.upper().str.strip()

# %% [markdown]
# ## 8. Feature engineering (useful for later EDA / ML / BI steps)

# %%
df["total_minutes"] = df["total_day_minutes"] + df["total_eve_minutes"] + df["total_night_minutes"] + df["total_intl_minutes"]
df["total_charge"] = df["total_day_charge"] + df["total_eve_charge"] + df["total_night_charge"] + df["total_intl_charge"]
df["total_calls"] = df["total_day_calls"] + df["total_eve_calls"] + df["total_night_calls"] + df["total_intl_calls"]
df["avg_charge_per_call"] = np.where(df["total_calls"] > 0, df["total_charge"] / df["total_calls"], 0)
df["has_intl_plan"] = (df["international_plan"] == "Yes").astype(int)
df["has_voicemail_plan"] = (df["voice_mail_plan"] == "Yes").astype(int)

# %% [markdown]
# ## 9. Normalization / Standardization (z-score) of numeric features
# We keep the *original* columns intact (needed for business-readable BI
# reports) and add `_zscore` companion columns for modeling, so nothing is
# lost either way.

# %%
from scipy.stats import zscore

scale_cols = ["account_length", "total_day_minutes", "total_eve_minutes",
              "total_night_minutes", "total_intl_minutes", "total_charge",
              "customer_service_calls"]
scale_cols = [c for c in scale_cols if c in df.columns]

for col in scale_cols:
    df[f"{col}_zscore"] = zscore(df[col])

# Min-max normalized version too (0-1 range), useful for some ML algorithms
for col in scale_cols:
    min_v, max_v = df[col].min(), df[col].max()
    df[f"{col}_norm"] = (df[col] - min_v) / (max_v - min_v) if max_v > min_v else 0.0

# %% [markdown]
# ## 10. Final structured vs unstructured note + data dictionary export

# %%
data_dictionary = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(t) for t in df.dtypes],
    "n_missing": df.isnull().sum().values,
    "n_unique": [df[c].nunique() for c in df.columns],
})
print(data_dictionary)

# %% [markdown]
# ## 11. Train/test split (mirrors the original BigML 80/20 provenance,
# but re-derived post-cleaning so both files reflect the cleaned data)

# %%
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["churn_flag"] if "churn_flag" in df.columns else None
)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# %% [markdown]
# ## 12. Save all outputs

# %%
df.to_csv(os.path.join(PROCESSED_DIR, "churn_cleaned.csv"), index=False)
train_df.to_csv(os.path.join(PROCESSED_DIR, "churn_train.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "churn_test.csv"), index=False)
data_dictionary.to_csv(os.path.join(PROCESSED_DIR, "data_dictionary.csv"), index=False)
outlier_report.to_csv(os.path.join(PROCESSED_DIR, "outlier_report.csv"))

print("Saved: churn_cleaned.csv, churn_train.csv, churn_test.csv, data_dictionary.csv, outlier_report.csv")
print("\nDone with Task: Data Collection & Cleaning.")
