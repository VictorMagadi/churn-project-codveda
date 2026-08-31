# %% [markdown]
# # Level 3 - Task 3: Strategic Decision Making with Prescriptive Analytics
# Codveda Business Analytics Internship
# Run 01, 04, and 05 first.
#
# Covers: simulation for business decisions, optimization (resource
# allocation), scenario analysis, and simple AI-driven recommendations —
# built on top of the churn-risk scores produced in script 05.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
VISUALS_DIR = os.path.join(PROJECT_ROOT, "visuals")

risk_df = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_risk_scores.csv"))
print(risk_df["risk_tier"].value_counts())

# %% [markdown]
# ## 1. Business problem
# We have a limited retention budget and want to allocate outreach
# (phone call vs email vs discount offer) across risk tiers to **maximize
# retained revenue** while staying within a fixed budget and staffing
# capacity. This is a classic resource-allocation / linear-programming
# problem.

# %%
tier_counts = risk_df["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
print(tier_counts)

# Assumptions (clearly labelled so they're easy to swap for real business figures)
avg_customer_monthly_value = 60          # $ average revenue per customer/month
retention_uplift = {"High": 0.35, "Medium": 0.18, "Low": 0.05}   # est. probability an outreach saves the customer
cost_per_outreach = {"High": 15, "Medium": 8, "Low": 3}          # $ cost of an intervention per tier
TOTAL_BUDGET = 5000                                              # $ available this cycle
MAX_STAFF_HOURS = 400                                             # staffing cap
HOURS_PER_OUTREACH = {"High": 0.5, "Medium": 0.25, "Low": 0.1}

# %% [markdown]
# ## 2. Linear programming — maximize expected retained value
# Decision variables: number of customers to contact in each tier (x_high, x_medium, x_low).
# Maximize: sum(uplift_i * value * x_i)   subject to budget & staffing caps & tier population caps.
# scipy.optimize.linprog minimizes by default, so we negate the objective.

# %%
tiers = ["High", "Medium", "Low"]
c = [-(retention_uplift[t] * avg_customer_monthly_value * 12) for t in tiers]  # negative = maximize annualized value

A_ub = [
    [cost_per_outreach[t] for t in tiers],          # budget constraint
    [HOURS_PER_OUTREACH[t] for t in tiers],          # staffing constraint
]
b_ub = [TOTAL_BUDGET, MAX_STAFF_HOURS]

bounds = [(0, tier_counts[t]) for t in tiers]   # can't contact more customers than exist in that tier

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

allocation = pd.Series(result.x, index=tiers).round().astype(int)
expected_value = -result.fun

print("Optimal outreach allocation (customers to contact per tier):")
print(allocation)
print(f"\nExpected annualized retained value: ${expected_value:,.0f}")
print(f"Budget used: ${sum(allocation[t]*cost_per_outreach[t] for t in tiers):,.0f} / ${TOTAL_BUDGET}")
print(f"Staff hours used: {sum(allocation[t]*HOURS_PER_OUTREACH[t] for t in tiers):.1f} / {MAX_STAFF_HOURS}")

allocation_df = allocation.rename("customers_to_contact").to_frame()
allocation_df["cost_per_customer"] = [cost_per_outreach[t] for t in tiers]
allocation_df["total_cost"] = allocation_df["customers_to_contact"] * allocation_df["cost_per_customer"]
allocation_df["retention_uplift_pct"] = [retention_uplift[t]*100 for t in tiers]
allocation_df.to_csv(os.path.join(PROCESSED_DIR, "optimal_retention_allocation.csv"))

# %% [markdown]
# ## 3. Scenario analysis — what if budget changes?

# %%
scenarios = {}
for budget in [2000, 5000, 8000, 12000, 20000]:
    res = linprog(c, A_ub=[[cost_per_outreach[t] for t in tiers], [HOURS_PER_OUTREACH[t] for t in tiers]],
                   b_ub=[budget, MAX_STAFF_HOURS], bounds=bounds, method="highs")
    scenarios[budget] = -res.fun if res.success else np.nan

scenario_df = pd.Series(scenarios, name="expected_retained_value").rename_axis("budget").reset_index()
print("\nScenario analysis (budget vs expected retained value):")
print(scenario_df)
scenario_df.to_csv(os.path.join(PROCESSED_DIR, "budget_scenario_analysis.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(scenario_df["budget"], scenario_df["expected_retained_value"], marker="o", color="#2E86AB")
ax.set_title("Scenario Analysis: Retention Budget vs Expected Value Saved")
ax.set_xlabel("Retention budget ($)")
ax.set_ylabel("Expected annualized retained value ($)")
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "13_budget_scenario_analysis.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. Monte Carlo simulation — uncertainty in the retention program's ROI
# The uplift assumptions above are estimates. Simulate ROI under uncertainty
# by sampling uplift rates from a distribution instead of fixed constants.

# %%
np.random.seed(42)
n_sims = 10000
sim_results = []
for _ in range(n_sims):
    sim_uplift = {
        "High": np.random.normal(0.35, 0.08),
        "Medium": np.random.normal(0.18, 0.05),
        "Low": np.random.normal(0.05, 0.02),
    }
    value = sum(
        allocation[t] * max(sim_uplift[t], 0) * avg_customer_monthly_value * 12
        for t in tiers
    )
    sim_results.append(value)

sim_results = np.array(sim_results)
print(f"\nMonte Carlo (n={n_sims}) — Expected retained value distribution:")
print(f"Mean: ${sim_results.mean():,.0f}, Median: ${np.median(sim_results):,.0f}")
print(f"5th percentile (pessimistic): ${np.percentile(sim_results, 5):,.0f}")
print(f"95th percentile (optimistic): ${np.percentile(sim_results, 95):,.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sim_results, bins=50, color="#2E86AB", alpha=0.8)
ax.axvline(np.median(sim_results), color="#E63946", linestyle="--", label="Median")
ax.set_title(f"Monte Carlo Simulation — Retention Program Value (n={n_sims})")
ax.set_xlabel("Expected annualized retained value ($)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, "14_monte_carlo_simulation.png"), dpi=150)
plt.close(fig)

# %% [markdown]
# ## 5. AI-driven recommendation summary (rule-based decision layer)
# Translates the analysis into plain-language, ranked action items — the
# kind of output a prescriptive-analytics dashboard would surface to a
# business decision-maker.

# %%
recommendations = []
for t in tiers:
    n_contact = allocation[t]
    if n_contact > 0:
        recommendations.append(
            f"Contact {n_contact} '{t}-risk' customers via "
            f"{'a phone call + discount offer' if t == 'High' else 'a targeted email + discount offer' if t == 'Medium' else 'an automated email nudge'}; "
            f"estimated {retention_uplift[t]*100:.0f}% chance of retaining each, "
            f"at ${cost_per_outreach[t]}/customer."
        )

recommendations.append(
    f"At the current ${TOTAL_BUDGET} budget, expected annualized value protected is "
    f"~${expected_value:,.0f} (Monte Carlo 90% interval: "
    f"${np.percentile(sim_results,5):,.0f} - ${np.percentile(sim_results,95):,.0f})."
)
if scenario_df["expected_retained_value"].iloc[-1] > scenario_df["expected_retained_value"].iloc[0] * 2:
    recommendations.append("Scenario analysis shows returns keep scaling well beyond the current budget — "
                            "consider requesting a larger retention budget next cycle.")

print("\n--- PRESCRIPTIVE RECOMMENDATIONS ---")
for i, r in enumerate(recommendations, 1):
    print(f"{i}. {r}")

with open(os.path.join(PROCESSED_DIR, "prescriptive_recommendations.txt"), "w") as f:
    f.write("PRESCRIPTIVE ANALYTICS — RETENTION STRATEGY RECOMMENDATIONS\n")
    f.write("=" * 60 + "\n\n")
    for i, r in enumerate(recommendations, 1):
        f.write(f"{i}. {r}\n\n")

print("\nSaved optimal_retention_allocation.csv, budget_scenario_analysis.csv, prescriptive_recommendations.txt")
print("Done with Task: Prescriptive Analytics & Strategic Decision Making.")
