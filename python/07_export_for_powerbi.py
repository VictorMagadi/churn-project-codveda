# %% [markdown]
# # Export a Power BI–ready Excel workbook
# Codveda Business Analytics Internship — Level 2, Task 1 (BI & Reporting)
#
# Run 01, 04, and 05 first. This bundles the cleaned data, model predictions,
# risk scores, and pre-aggregated summary tables into ONE multi-sheet .xlsx
# file that Power BI can import directly (Get Data -> Excel -> pick sheets).

# %%
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

cleaned = pd.read_csv(os.path.join(PROCESSED_DIR, "churn_cleaned.csv"))
predictions = pd.read_csv(os.path.join(PROCESSED_DIR, "model_predictions.csv"))
risk_scores = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_risk_scores.csv"))

# %% [markdown]
# ## Build pre-aggregated summary tables for fast dashboard visuals

# %%
by_state = cleaned.groupby("state").agg(
    customers=("state", "count"),
    churned=("churn_flag", "sum"),
    churn_rate_pct=("churn_flag", lambda s: round(s.mean() * 100, 2)),
    avg_day_minutes=("total_day_minutes", "mean"),
    avg_total_charge=("total_charge", "mean"),
    avg_service_calls=("customer_service_calls", "mean"),
).reset_index().sort_values("churn_rate_pct", ascending=False)

by_plan = cleaned.groupby(["international_plan", "voice_mail_plan"]).agg(
    customers=("state", "count"),
    churn_rate_pct=("churn_flag", lambda s: round(s.mean() * 100, 2)),
).reset_index()

kpi_summary = pd.DataFrame({
    "metric": [
        "Total Customers", "Total Churned", "Overall Churn Rate (%)",
        "Avg Monthly Charge ($)", "Avg Customer Service Calls",
        "Customers on International Plan (%)", "Customers on Voicemail Plan (%)",
    ],
    "value": [
        len(cleaned),
        int(cleaned["churn_flag"].sum()),
        round(cleaned["churn_flag"].mean() * 100, 2),
        round(cleaned["total_charge"].mean(), 2),
        round(cleaned["customer_service_calls"].mean(), 2),
        round((cleaned["international_plan"] == "Yes").mean() * 100, 2),
        round((cleaned["voice_mail_plan"] == "Yes").mean() * 100, 2),
    ],
})

# %% [markdown]
# ## Write the multi-sheet workbook

# %%
out_path = os.path.join(PROCESSED_DIR, "churn_for_powerbi.xlsx")
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    kpi_summary.to_excel(writer, sheet_name="KPI_Summary", index=False)
    cleaned.to_excel(writer, sheet_name="Customer_Detail", index=False)
    predictions.to_excel(writer, sheet_name="Churn_Predictions", index=False)
    risk_scores.to_excel(writer, sheet_name="Risk_Scores", index=False)
    by_state.to_excel(writer, sheet_name="By_State", index=False)
    by_plan.to_excel(writer, sheet_name="By_Plan_Type", index=False)

print(f"Saved Power BI–ready workbook: {out_path}")
print("Sheets: KPI_Summary, Customer_Detail, Churn_Predictions, Risk_Scores, By_State, By_Plan_Type")
