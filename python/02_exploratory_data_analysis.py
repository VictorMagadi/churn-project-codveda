# %% [markdown]
# # Level 1 - Task 3: Exploratory Data Analysis (EDA)
# Codveda Business Analytics Internship
# Run 01_data_collection_cleaning.py first.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless/script runs; Jupyter/Spyder will still show plots inline when run interactively
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")
os.makedirs(VISUALS_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, "churn_cleaned.csv"))
print(df.shape)
df.head()

# %% [markdown]
# ## 1. Statistical summaries (mean, median, mode, std)

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
summary = df[numeric_cols].agg(["mean", "median", "std", "min", "max"]).T
summary["mode"] = [df[c].mode().iloc[0] for c in numeric_cols]
summary = summary[["mean", "median", "mode", "std", "min", "max"]]
print(summary)
summary.to_csv(os.path.join(PROCESSED_DIR, "eda_summary_stats.csv"))

# %% [markdown]
# ## 2. Churn rate overview (the core business KPI)

# %%
churn_rate = df["churn_flag"].mean() * 100
print(f"Overall churn rate: {churn_rate:.2f}%")

churn_by_state = df.groupby("state")["churn_flag"].mean().sort_values(ascending=False) * 100
print("\nTop 10 states by churn rate:")
print(churn_by_state.head(10))

# %% [markdown]
# ## 3. Visualization — churn distribution

# %%
fig, ax = plt.subplots(figsize=(5, 4))
df["churn"].value_counts().plot(kind="bar", color=["#2E86AB", "#E63946"], ax=ax)
ax.set_title("Customer Churn Distribution")
ax.set_xlabel("Churned")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "01_churn_distribution.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. Histograms of key usage metrics

# %%
hist_cols = ["total_day_minutes", "total_eve_minutes", "total_night_minutes",
             "total_intl_minutes", "customer_service_calls", "account_length"]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.flatten(), hist_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#2E86AB")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "02_histograms_key_metrics.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 5. Scatter plots — usage vs charge relationships

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
pairs = [("total_day_minutes", "total_day_charge"),
         ("total_eve_minutes", "total_eve_charge"),
         ("total_night_minutes", "total_night_charge")]
for ax, (x, y) in zip(axes, pairs):
    sns.scatterplot(data=df, x=x, y=y, hue="churn", alpha=0.5, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title(f"{x} vs {y}")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "03_scatter_usage_vs_charge.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 6. Correlation & regression-style insight (correlation heatmap)

# %%
corr_cols = ["account_length", "total_day_minutes", "total_eve_minutes", "total_night_minutes",
             "total_intl_minutes", "customer_service_calls", "total_charge", "churn_flag"]
corr_matrix = df[corr_cols].corr()
print(corr_matrix["churn_flag"].sort_values(ascending=False))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix — Usage Metrics vs Churn")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "04_correlation_heatmap.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 7. Churn rate by categorical drivers (plan type, service calls)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
df.groupby("international_plan")["churn_flag"].mean().mul(100).plot(kind="bar", ax=axes[0], color="#2E86AB")
axes[0].set_title("Churn Rate by International Plan (%)")
axes[0].set_ylabel("Churn rate %")

df.groupby("customer_service_calls")["churn_flag"].mean().mul(100).plot(kind="bar", ax=axes[1], color="#E63946")
axes[1].set_title("Churn Rate by # Customer Service Calls (%)")
axes[1].set_ylabel("Churn rate %")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "05_churn_by_plan_and_calls.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 8. Trend/seasonality proxy — churn vs account tenure (account_length)
# This dataset has no explicit date field, so we use `account_length`
# (customer tenure in days) as the time-axis proxy for a trend view.

# %%
df["tenure_bucket"] = pd.cut(df["account_length"], bins=10)
tenure_churn = df.groupby("tenure_bucket", observed=True)["churn_flag"].mean() * 100

fig, ax = plt.subplots(figsize=(9, 4.5))
tenure_churn.plot(kind="line", marker="o", ax=ax, color="#E63946")
ax.set_title("Churn Rate Across Customer Tenure Buckets")
ax.set_ylabel("Churn rate %")
ax.set_xlabel("Account length bucket (days)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "06_churn_trend_by_tenure.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 9. Anomaly spotting via boxplots (post-capping sanity check)

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ["total_day_minutes", "total_eve_minutes", "customer_service_calls"]):
    sns.boxplot(data=df, x="churn", y=col, hue="churn", legend=False, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title(f"{col} by churn status")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "07_boxplots_by_churn.png"), dpi=150)
plt.close(fig)

print(f"\nAll EDA visuals saved to: {VISUALS_DIR}")
print("Done with Task: Exploratory Data Analysis.")
