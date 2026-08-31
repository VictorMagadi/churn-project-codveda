# Codveda Technology — Business Analytics Internship Project
### Full-stack solution: Python + R + SQL + Power BI

This folder is a complete, run-anywhere implementation of **all 9 tasks** across
the 3 levels of the Codveda Business Analytics task list, built on the
**Telecom Customer Churn dataset** (`churn-bigml-80.csv` = training/analysis set,
`churn-bigml-20.csv` = holdout/validation set — this is the classic BigML churn
dataset split, so we use it exactly as intended: 80% to build, 20% to test).

Even though the internship only requires **2 of 3 tasks per level**, every
single task is completed here so you can pick whichever two you like per level
and still submit polished, working proof for the rest.

---

## 1. Folder Map

```
Codveda_Business_Analytics_Project/
│
├── README.md                          <- you are here
├── data/
│   ├── raw/                           <- original CSVs, untouched
│   │   ├── churn-bigml-80.csv
│   │   └── churn-bigml-20.csv
│   └── processed/                     <- generated automatically by the scripts
│       ├── churn_cleaned.csv          <- cleaned, standardized, feature-engineered
│       ├── churn_train.csv / churn_test.csv
│       ├── eda_summary_stats.csv
│       ├── statistical_test_results.csv
│       ├── model_predictions.csv
│       ├── customer_risk_scores.csv
│       └── churn_for_powerbi.xlsx     <- ready-made Power BI data source
│
├── python/                            <- run in Jupyter Notebook OR Spyder
│   ├── 00_run_all.py
│   ├── 01_data_collection_cleaning.py     (Level 1, Task 2)
│   ├── 02_exploratory_data_analysis.py    (Level 1, Task 3)
│   ├── 03_statistical_analysis.py         (Level 2, Task 3)
│   ├── 04_predictive_analytics_ml.py      (Level 3, Task 1)
│   ├── 05_risk_fraud_anomaly_detection.py (Level 3, Task 2)
│   └── 06_prescriptive_analytics.py       (Level 3, Task 3)
│
├── r/                                  <- run in RStudio
│   ├── 01_data_cleaning.R
│   ├── 02_eda.R
│   ├── 03_statistical_analysis.R
│   ├── 04_predictive_modeling.R
│   └── 05_prescriptive_and_risk.R
│
├── sql/                                 (Level 2, Task 2)
│   ├── 01_create_tables_sqlite.sql
│   ├── 02_business_analytics_queries.sql
│   └── run_sql_demo.py                <- loads CSV into SQLite and runs the .sql file for you
│
├── powerbi/                              (Level 2, Task 1)
│   ├── PowerBI_Setup_Guide.md
│   └── DAX_Measures.txt
│
├── docs/
│   ├── Level1_Task1_Understanding_Business_Analytics.md
│   └── Project_Report.md
│
└── visuals/                            <- charts auto-saved here when you run the Python scripts
```

## 2. How each tool is used

| Layer | Tool | What it covers |
|---|---|---|
| **Python** | Jupyter Notebook & Spyder | Cleaning, EDA, statistics, ML (regression/classification/clustering), anomaly detection, optimization |
| **R** | RStudio | Parallel implementation of cleaning, EDA, stats, and modeling (`tidyverse`, `caret`, `ggplot2`) |
| **SQL** | SQLite (via Python, no server needed) | Aggregation, joins, window functions, business KPI queries |
| **Power BI** | Power BI Desktop | Interactive dashboard from the cleaned Excel export, with DAX measures provided |

## 3. Running the Python scripts

**They work identically in Jupyter Notebook and Spyder** because every script
uses the universal cell-separator `# %%`, which both Jupyter (via
`jupytext`/"Convert to notebook" or just running cell-by-cell) and Spyder
recognize as a runnable cell. You can also just run them top-to-bottom as
plain `.py` scripts from a terminal.

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels openpyxl xgboost
```

- **Jupyter**: open each `.py` in Jupyter (`jupyter notebook`) — Jupyter treats
  `# %%` as a cell boundary automatically in JupyterLab ≥3, or right-click →
  "Open as Notebook", or run `jupytext --to notebook 01_data_collection_cleaning.py`
  to get a real `.ipynb`.
- **Spyder**: open the file, cells separated by `# %%` show up automatically —
  use Ctrl+Enter to run cell by cell.
- **Terminal**: `python 01_data_collection_cleaning.py`

Run them **in order (01 → 06)** the first time, since later scripts read the
`data/processed/` files created by earlier ones. Or just run `00_run_all.py`.

## 4. Running the R scripts (RStudio)

```r
install.packages(c("tidyverse","caret","corrplot","e1071","cluster",
                    "factoextra","rpart","rpart.plot","openxlsx"))
```
Open `r/01_data_cleaning.R` in RStudio and run top to bottom (Ctrl+Alt+R),
then 02, 03, 04, 05 in order — same dependency chain as the Python side.

## 5. SQL

No database server needed. `sql/run_sql_demo.py` spins up a local SQLite file
(`data/processed/churn.db`), loads the cleaned CSV into it, and executes every
query in `02_business_analytics_queries.sql`, printing results. You can also
open `churn.db` directly in DB Browser for SQLite, or copy the `.sql` files
into MySQL/PostgreSQL/SQL Server with minor syntax tweaks (noted in the file).

## 6. Power BI

1. Run the Python pipeline once (steps 01–05) so `data/processed/churn_for_powerbi.xlsx` exists.
2. Open Power BI Desktop → Get Data → Excel → select that file.


