# %% [markdown]
# # Level 3 - Task 1: Predictive Analytics & Machine Learning
# Codveda Business Analytics Internship
# Run 01_data_collection_cleaning.py first.
#
# Covers: regression (forecast usage/spend), classification (churn
# prediction), clustering (customer segmentation), and a simple "deployment"
# function that scores new customers.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, silhouette_score
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")
MODELS_DIR = os.path.join(PROJECT_ROOT, "python", "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)

train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "churn_train.csv"))
test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "churn_test.csv"))
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

feature_cols = [
    "account_length", "total_day_minutes", "total_day_calls",
    "total_eve_minutes", "total_eve_calls",
    "total_night_minutes", "total_night_calls",
    "total_intl_minutes", "total_intl_calls",
    "customer_service_calls", "has_intl_plan", "has_voicemail_plan",
    "number_vmail_messages",
]
feature_cols = [c for c in feature_cols if c in train_df.columns]

# %% [markdown]
# ## 1. Regression model — forecast total monthly charge (demand/spend forecasting)

# %%
reg_target = "total_charge"
X_train_r = train_df[feature_cols]
y_train_r = train_df[reg_target]
X_test_r = test_df[feature_cols]
y_test_r = test_df[reg_target]

lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
pred_lin = lin_reg.predict(X_test_r)

rf_reg = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10)
rf_reg.fit(X_train_r, y_train_r)
pred_rf = rf_reg.predict(X_test_r)

for name, preds in [("Linear Regression", pred_lin), ("Random Forest Regressor", pred_rf)]:
    rmse = mean_squared_error(y_test_r, preds) ** 0.5
    r2 = r2_score(y_test_r, preds)
    print(f"[{name}] RMSE={rmse:.3f}, R2={r2:.4f}")

joblib.dump(rf_reg, os.path.join(MODELS_DIR, "regression_total_charge_rf.pkl"))

# %% [markdown]
# ## 2. Classification models — customer churn prediction

# %%
clf_target = "churn_flag"
X_train_c = train_df[feature_cols]
y_train_c = train_df[clf_target]
X_test_c = test_df[feature_cols]
y_test_c = test_df[clf_target]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_c)
X_test_scaled = scaler.transform(X_test_c)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train_c)
pred_log = log_reg.predict(X_test_scaled)
proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

rf_clf = RandomForestClassifier(n_estimators=400, random_state=42, max_depth=12, class_weight="balanced")
rf_clf.fit(X_train_c, y_train_c)
pred_rf_clf = rf_clf.predict(X_test_c)
proba_rf_clf = rf_clf.predict_proba(X_test_c)[:, 1]

clf_results = []
for name, preds, proba in [("Logistic Regression", pred_log, proba_log),
                            ("Random Forest Classifier", pred_rf_clf, proba_rf_clf)]:
    acc = accuracy_score(y_test_c, preds)
    prec = precision_score(y_test_c, preds)
    rec = recall_score(y_test_c, preds)
    f1 = f1_score(y_test_c, preds)
    auc = roc_auc_score(y_test_c, proba)
    clf_results.append((name, acc, prec, rec, f1, auc))
    print(f"\n[{name}] Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f} ROC-AUC={auc:.4f}")
    print(classification_report(y_test_c, preds, target_names=["Stayed", "Churned"]))

clf_results_df = pd.DataFrame(clf_results, columns=["model", "accuracy", "precision", "recall", "f1", "roc_auc"])
clf_results_df.to_csv(os.path.join(PROCESSED_DIR, "classification_model_comparison.csv"), index=False)

joblib.dump(rf_clf, os.path.join(MODELS_DIR, "churn_classifier_rf.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.pkl"))

# %% [markdown]
# ### Confusion matrix + ROC curve (best model = Random Forest)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cm = confusion_matrix(y_test_c, pred_rf_clf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stayed", "Churned"],
            yticklabels=["Stayed", "Churned"], ax=axes[0])
axes[0].set_title("Confusion Matrix — Random Forest Churn Classifier")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

fpr, tpr, _ = roc_curve(y_test_c, proba_rf_clf)
axes[1].plot(fpr, tpr, color="#E63946", label=f"RF (AUC={roc_auc_score(y_test_c, proba_rf_clf):.3f})")
fpr2, tpr2, _ = roc_curve(y_test_c, proba_log)
axes[1].plot(fpr2, tpr2, color="#2E86AB", label=f"LogReg (AUC={roc_auc_score(y_test_c, proba_log):.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_title("ROC Curve — Churn Classifiers")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "08_classification_performance.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ### Feature importance (business-readable churn drivers)

# %%
importances = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)

fig, ax = plt.subplots(figsize=(9, 6))
importances.plot(kind="barh", ax=ax, color="#2E86AB")
ax.invert_yaxis()
ax.set_title("Feature Importance — Churn Prediction (Random Forest)")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "09_feature_importance.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Clustering — customer/market segmentation (KMeans)

# %%
cluster_features = ["total_day_minutes", "total_eve_minutes", "total_night_minutes",
                     "total_intl_minutes", "customer_service_calls", "account_length"]
full_df = pd.concat([train_df, test_df], ignore_index=True)
X_cluster = full_df[cluster_features]
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

inertias, silhouettes = [], []
k_range = range(2, 8)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_cluster_scaled, labels))

best_k = list(k_range)[int(np.argmax(silhouettes))]
print(f"Best k by silhouette score: {best_k}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#2E86AB")
axes[0].set_title("Elbow Method (Inertia)")
axes[0].set_xlabel("k")
axes[1].plot(list(k_range), silhouettes, marker="o", color="#E63946")
axes[1].set_title("Silhouette Score by k")
axes[1].set_xlabel("k")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "10_clustering_selection.png"), dpi=150)
plt.close(fig)

final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
full_df["customer_segment"] = final_km.fit_predict(X_cluster_scaled)

segment_profile = full_df.groupby("customer_segment")[cluster_features + ["churn_flag"]].mean()
print("\nSegment profiles:")
print(segment_profile)
segment_profile.to_csv(os.path.join(PROCESSED_DIR, "customer_segment_profiles.csv"))

joblib.dump(final_km, os.path.join(MODELS_DIR, "customer_segmentation_kmeans.pkl"))

# %% [markdown]
# ## 4. "Deployment" — score every customer with churn probability + segment
# This is the artifact a business team / Power BI dashboard would consume.

# %%
full_df["predicted_churn_probability"] = rf_clf.predict_proba(full_df[feature_cols])[:, 1]
full_df["predicted_churn_risk_tier"] = pd.cut(
    full_df["predicted_churn_probability"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low", "Medium", "High"],
)

output_cols = ["customer_id", "state", "account_length", "churn", "churn_flag",
                "customer_segment", "predicted_churn_probability", "predicted_churn_risk_tier"]
full_df[output_cols].to_csv(os.path.join(PROCESSED_DIR, "model_predictions.csv"), index=False)

print(full_df["predicted_churn_risk_tier"].value_counts())
print("\nSaved model_predictions.csv, customer_segment_profiles.csv, classification_model_comparison.csv")
print("Saved trained models to python/saved_models/")
print("Done with Task: Predictive Analytics & Machine Learning.")
