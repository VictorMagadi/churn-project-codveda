# Level 1 – Task 1: Understanding Business Analytics

This task is conceptual (no dataset required), so it's provided as a
reference write-up rather than code — worth keeping for your submission
notes or a LinkedIn post explaining what you learned.

## 1. The Four Types of Analytics

| Type | Question it answers | Example in this project |
|---|---|---|
| **Descriptive** | "What happened?" | `02_exploratory_data_analysis` — churn rate, usage summaries, distributions |
| **Diagnostic** | "Why did it happen?" | `03_statistical_analysis` — t-tests/chi-square isolating drivers like service calls and international plan |
| **Predictive** | "What will happen?" | `04_predictive_analytics_ml` — churn probability model, usage forecasting |
| **Prescriptive** | "What should we do about it?" | `06_prescriptive_analytics` — optimized retention-budget allocation |

## 2. Role of Analytics in Business Decision-Making

Analytics moves decisions from intuition to evidence. In this project, that
looks like: instead of guessing which customers might leave, the churn model
gives every customer a probability score; instead of spreading a retention
budget evenly, the optimization step (Level 3, Task 3) allocates it toward
the customers where a dollar of outreach saves the most expected revenue.

Good business analytics practice generally follows a loop:
**collect → clean → explore → explain → predict → prescribe → act → measure
→ repeat.** Every one of those stages has a corresponding script in this
project folder.

## 3. Key Business Metrics & KPIs (as used here)

- **Churn rate** — % of customers who left in the period. The single most
  important KPI for a subscription/telecom business.
- **Customer service calls** — an early-warning KPI; higher call volume
  strongly correlates with churn in this dataset.
- **Average revenue per customer (ARPU)** — modeled here as
  `total_charge`.
- **Retention uplift / cost-to-serve** — used in the prescriptive model to
  judge whether an intervention is worth its cost.
- **Risk score / risk tier** — a composite KPI blending anomaly detection,
  rule-based flags, and predicted churn probability into one actionable
  number.

## 4. Industries Where Business Analytics Is Applied

- **Telecom** (this project's dataset) — churn prediction, network usage
  optimization, fraud/SIM-box detection.
- **Retail & e-commerce** — demand forecasting, price optimization,
  recommendation engines.
- **Banking & finance** — credit risk scoring, fraud detection, algorithmic
  trading.
- **Healthcare** — patient risk stratification, hospital resource
  scheduling.
- **Manufacturing & logistics** — predictive maintenance, supply-chain
  optimization.
- **Marketing** — A/B testing, customer segmentation, attribution modeling.

## 5. Tools Commonly Used (and used in this project)

- **Python** (pandas, scikit-learn, statsmodels) — flexible scripting,
  machine learning, automation.
- **R** (tidyverse, caret) — statistical rigor, strong for hypothesis
  testing and academic-style analysis.
- **SQL** — the universal language for pulling and aggregating data out of
  operational databases.
- **Power BI / Tableau** — the layer that turns analysis into a shareable,
  interactive story for non-technical stakeholders.
