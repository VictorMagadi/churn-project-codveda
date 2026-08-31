# %% [markdown]
# # Level 3 - Task 2: Risk Analysis & Fraud Detection
# Codveda Business Analytics Internship
# Run 01 and 04 first (uses cleaned data + model predictions).
#
# The churn dataset has no labelled "fraud" field, so we apply the same
# unsupervised anomaly-detection techniques a fraud team would use — flagging
# statistically unusual telecom usage patterns (e.g. abnormal call volume /
# international usage spikes that resemble SIM-boxing or account takeover
# patterns in telecom fraud) — and combine it with churn risk into one
# overall business risk score.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "churn_cleaned.csv"))
predictions = pd.read_csv(os.path.join(PROCESSED_DIR, "model_predictions.csv"))

# %% [markdown]
# ## 1. Anomaly detection in "transaction-like" usage data
# Treat each customer's usage record as a transaction. Flag statistical
# anomalies using Isolation Forest — the standard unsupervised algorithm for
# fraud/anomaly detection when there are no labelled fraud cases.

# %%
anomaly_features = [
    "total_day_minutes", "total_day_calls", "total_day_charge",
    "total_eve_minutes", "total_eve_calls",
    "total_night_minutes", "total_night_calls",
    "total_intl_minutes", "total_intl_calls", "total_intl_charge",
    "customer_service_calls",
]
X = df[anomaly_features]
X_scaled = StandardScaler().fit_transform(X)

iso_forest = IsolationForest(n_estimators=300, contamination=0.03, random_state=42)
df["anomaly_flag_iforest"] = iso_forest.fit_predict(X_scaled)   # -1 = anomaly, 1 = normal
df["anomaly_score_iforest"] = -iso_forest.score_samples(X_scaled)  # higher = more anomalous

n_anomalies = (df["anomaly_flag_iforest"] == -1).sum()
print(f"Isolation Forest flagged {n_anomalies} anomalous records ({n_anomalies/len(df)*100:.2f}%)")

# %% [markdown]
# ## 2. Cross-check with One-Class SVM (second method for robustness)

# %%
oc_svm = OneClassSVM(nu=0.03, kernel="rbf", gamma="scale")
df["anomaly_flag_svm"] = oc_svm.fit_predict(X_scaled)

agreement = ((df["anomaly_flag_iforest"] == -1) & (df["anomaly_flag_svm"] == -1)).sum()
print(f"Records flagged as anomalous by BOTH methods (higher confidence fraud/risk candidates): {agreement}")
df["high_confidence_anomaly"] = ((df["anomaly_flag_iforest"] == -1) & (df["anomaly_flag_svm"] == -1)).astype(int)

# %% [markdown]
# ## 3. Rule-based risk indicators (classic fraud-analytics heuristics)
# Combine statistical anomaly detection with simple business rules — this is
# how real fraud teams triangulate: unsupervised model + domain rules.

# %%
df["rule_excessive_intl_usage"] = (df["total_intl_calls"] > df["total_intl_calls"].quantile(0.98)).astype(int)
df["rule_excessive_service_calls"] = (df["customer_service_calls"] >= 6).astype(int)
df["rule_unusual_charge_ratio"] = (
    df["avg_charge_per_call"] > df["avg_charge_per_call"].quantile(0.98)
).astype(int)

df["risk_rule_score"] = (
    df["rule_excessive_intl_usage"] + df["rule_excessive_service_calls"] + df["rule_unusual_charge_ratio"]
)

# %% [markdown]
# ## 4. Composite business risk score (0-100)
# Combines: (a) statistical anomaly score, (b) rule-based flags, and (c) the
# churn probability from the predictive model — a single number ops teams
# can sort by to prioritize outreach / investigation.

# %%
df = df.merge(predictions[["customer_id", "predicted_churn_probability"]],
               on="customer_id", how="left")

def minmax(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

df["risk_score"] = (
    0.4 * minmax(df["anomaly_score_iforest"]) +
    0.3 * minmax(df["risk_rule_score"]) +
    0.3 * df["predicted_churn_probability"].fillna(df["predicted_churn_probability"].mean())
) * 100

df["risk_tier"] = pd.cut(df["risk_score"], bins=[-1, 33, 66, 100], labels=["Low", "Medium", "High"])

print(df["risk_tier"].value_counts())
print("\nTop 10 highest-risk customers:")
print(df.sort_values("risk_score", ascending=False)[
    ["state", "account_length", "customer_service_calls", "total_intl_calls",
     "high_confidence_anomaly", "predicted_churn_probability", "risk_score", "risk_tier"]
].head(10))

# %% [markdown]
# ## 5. Visualizations

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df["risk_score"], bins=30, kde=True, color="#E63946", ax=axes[0])
axes[0].set_title("Distribution of Composite Risk Scores")

df["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).plot(
    kind="bar", ax=axes[1], color=["#2E86AB", "#F4A261", "#E63946"]
)
axes[1].set_title("Customers by Risk Tier")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "11_risk_score_distribution.png"), dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df, x="total_intl_calls", y="customer_service_calls",
                 hue=df["anomaly_flag_iforest"].map({1: "Normal", -1: "Anomaly"}),
                 palette={"Normal": "#2E86AB", "Anomaly": "#E63946"}, alpha=0.6, ax=ax)
ax.set_title("Anomaly Detection: Intl Calls vs Service Calls")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "12_anomaly_scatter.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 6. Save outputs

# %%
risk_cols = [
    "customer_id", "state", "account_length", "churn", "customer_service_calls",
    "total_intl_calls", "avg_charge_per_call",
    "anomaly_flag_iforest", "anomaly_flag_svm", "high_confidence_anomaly",
    "risk_rule_score", "predicted_churn_probability", "risk_score", "risk_tier",
]
df[risk_cols].to_csv(os.path.join(PROCESSED_DIR, "customer_risk_scores.csv"), index=False)
print("\nSaved customer_risk_scores.csv")
print("Done with Task: Risk Analysis & Fraud Detection.")
