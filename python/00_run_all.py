# %% [markdown]
# # Run the entire Python pipeline end-to-end
# Executes scripts 01 through 07 in order. Use this for a one-click run in


# %%
import os
import runpy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

pipeline = [
    "01_data_collection_cleaning.py",
    "02_exploratory_data_analysis.py",
    "03_statistical_analysis.py",
    "04_predictive_analytics_ml.py",
    "05_risk_fraud_anomaly_detection.py",
    "06_prescriptive_analytics.py",
    "07_export_for_powerbi.py",
]

for script in pipeline:
    path = os.path.join(SCRIPT_DIR, script)
    print("\n" + "=" * 100)
    print(f"RUNNING: {script}")
    print("=" * 70)
    runpy.run_path(path, run_name="__main__")

print("\n\nPipeline complete. Check data/processed/, visuals/, and python/saved_models/.")
