# Codveda Business Analytics Internship — Project Report
### Telecom Customer Churn Analysis (Python + R + SQL + Power BI)

**Dataset:** BigML Telecom Churn (`churn-bigml-80.csv` + `churn-bigml-20.csv`),
3,333 customers after cleaning and de-duplication, 20 original fields across
51 U.S. states/DC, target: `churn` (True/False).

---

## 1. Data Cleaning (Level 1, Task 2)
- 0 missing values in the raw data (pipeline still includes defensive
  median/mode imputation for production reuse).
- 2 exact duplicate rows removed.
- Outliers in usage/spend columns (day/eve/night/intl minutes & charges,
  service calls) capped at IQR fences rather than deleted, to avoid biasing
  the churn signal from real high-usage customers.
- Added engineered features: `total_minutes`, `total_charge`, `total_calls`,
  `avg_charge_per_call`, plan-adoption flags, z-score and min-max scaled
  versions of key numeric fields.

## 2. Exploratory Data Analysis (Level 1, Task 3)
- **Overall churn rate: 14.49%.**
- Highest-churn states: **CA and NJ (26.47%)**, TX (25.0%), MD (24.3%).
- Correlation with churn (strongest first): `total_charge` (r=0.23),
  `total_day_minutes` (r=0.20), `customer_service_calls` (r=0.15).
- Full chart set saved in `visuals/` (churn distribution, histograms,
  scatter plots, correlation heatmap, churn-by-plan, tenure trend, boxplots).

## 3. Statistical Analysis (Level 2, Task 3)

| Test | Statistic | p-value | Conclusion |
|---|---|---|---|
| T-test: day minutes, churn vs stay | t = 9.70 | < 0.0001 | Churners use significantly more daytime minutes |
| T-test: service calls, churn vs stay | t = 7.59 | < 0.0001 | Churners call support significantly more |
| Chi-square: intl plan vs churn | χ² = 222.6 | < 0.0001 | International plan strongly associated with churn |
| Chi-square: voicemail plan vs churn | χ² = 34.1 | < 0.0001 | Voicemail plan associated with **lower** churn |
| 95% CI, churn rate | — | — | 13.30% – 15.69% (±1.20 pp) |
| P(service calls > 4), Poisson(λ=1.48) | — | 0.0175 | ~1.75% baseline risk of a customer escalating heavily |
| A/B test: simulated retention campaign | z = 4.26 | < 0.0001 | A modeled 12pp uplift campaign is statistically detectable at n=150/arm |

Customers on the **international plan churn at 43.7%** vs **13.9%** for
those without one — the single strongest churn signal in the dataset, and a
clear target for plan-pricing review.

## 4. Predictive Analytics & ML (Level 3, Task 1)

**Regression** (forecasting `total_charge` from usage): Random Forest
achieved R² = 0.987 (near-perfect, expected — billing is a near-linear
function of minutes in this dataset).

**Classification** (churn prediction):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 74.8% | 33.0% | 71.1% | 0.451 | 0.802 |
| **Random Forest** | **94.0%** | **96.7%** | 60.8% | 0.747 | **0.944** |

Random Forest is the stronger overall model (much higher precision and
AUC); Logistic Regression trades precision for higher recall, which can be
preferable if the business would rather over-flag than under-flag at-risk
customers. Top churn predictors: `total_day_minutes`,
`customer_service_calls`, `has_intl_plan`, `total_eve_minutes`.

**Clustering**: Silhouette analysis selected **k=2** customer segments; the
higher-churn segment (18.5% churn) skews toward slightly higher day-minute
usage than the lower-churn segment (11.4% churn).

## 5. Risk Analysis & Fraud/Anomaly Detection (Level 3, Task 2)
- Isolation Forest + One-Class SVM cross-validated anomaly flags: **100**
  statistically anomalous usage records (3% contamination assumption), with
  **53** flagged by both methods (high-confidence candidates for manual
  review).
- Composite risk score (40% anomaly score + 30% rule-based flags + 30%
  churn probability) segments customers into: **2,878 Low / 421 Medium / 34
  High** risk.
- Full customer-level risk scores exported to
  `data/processed/customer_risk_scores.csv`.

## 6. Prescriptive Analytics (Level 3, Task 3)
Linear-programming optimization of a **$5,000** retention budget under a
400-staff-hour cap:

| Tier | Customers contacted | Cost/customer | Total cost | Est. retention uplift |
|---|---|---|---|---|
| High | 34 | $15 | $510 | 35% |
| Medium | 421 | $8 | $3,368 | 18% |
| Low | 374 | $3 | $1,122 | 5% |

**Expected annualized retained value: ~$76,600** (Monte Carlo 90% interval:
$50,200 – $103,200, accounting for uncertainty in the assumed uplift rates).
Scenario analysis shows returns keep scaling well past $5,000 — a case for
requesting a larger retention budget next cycle if these assumptions hold up
in a real pilot.

## 7. Business Recommendations
1. **Investigate the international plan's pricing/experience** — it's the
   single strongest churn driver (43.7% vs 13.9% churn rate).
2. **Treat 4+ customer service calls as a hard escalation trigger** —
   churners call support significantly more, and it's a cheap, real-time
   signal to act on before renewal.
3. **Prioritize outreach using the composite risk score**, not churn
   probability alone — it also catches anomalous usage patterns that a
   churn-only model would miss.
4. **Pilot the $5,000 retention allocation** proposed in Section 6, then
   re-run the Monte Carlo simulation with *actual* observed uplift to
   refine the model before scaling the budget.

---
*All numbers above are reproducible by running the Python pipeline
(`python/00_run_all.py`) end-to-end against `data/raw/`.*
