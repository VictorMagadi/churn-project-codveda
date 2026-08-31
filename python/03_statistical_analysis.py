# %% [markdown]
# # Level 2 - Task 3: Statistical Analysis for Business Decisions
# Codveda Business Analytics Internship
# Run 01_data_collection_cleaning.py first.
#
# Covers: hypothesis testing (t-test, chi-square), probability distributions
# for risk, A/B test simulation for a marketing campaign, and confidence
# intervals / margin of error.

# %%
import os
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "churn_cleaned.csv"))
results = []  # collect (test_name, statistic, p_value, conclusion) for export

# %% [markdown]
# ## 1. Two-sample t-test
# H0: churners and non-churners have the same mean `total_day_minutes`.
# H1: they differ. (Business question: does heavy daytime usage relate to churn?)

# %%
churn_minutes = df.loc[df["churn_flag"] == 1, "total_day_minutes"]
stay_minutes = df.loc[df["churn_flag"] == 0, "total_day_minutes"]

t_stat, p_val = stats.ttest_ind(churn_minutes, stay_minutes, equal_var=False)
conclusion = "Reject H0: means differ significantly" if p_val < 0.05 else "Fail to reject H0"
print(f"T-test (total_day_minutes, churn vs stay): t={t_stat:.3f}, p={p_val:.5f} -> {conclusion}")
results.append(("t-test: day minutes churn vs stay", t_stat, p_val, conclusion))

# Second t-test on customer_service_calls (common churn driver)
churn_calls = df.loc[df["churn_flag"] == 1, "customer_service_calls"]
stay_calls = df.loc[df["churn_flag"] == 0, "customer_service_calls"]
t_stat2, p_val2 = stats.ttest_ind(churn_calls, stay_calls, equal_var=False)
conclusion2 = "Reject H0: means differ significantly" if p_val2 < 0.05 else "Fail to reject H0"
print(f"T-test (customer_service_calls, churn vs stay): t={t_stat2:.3f}, p={p_val2:.5f} -> {conclusion2}")
results.append(("t-test: service calls churn vs stay", t_stat2, p_val2, conclusion2))

# %% [markdown]
# ## 2. Chi-square test of independence
# H0: international plan subscription is independent of churn.

# %%
contingency = pd.crosstab(df["international_plan"], df["churn"])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
conclusion_chi = "Reject H0: variables are dependent (associated)" if p_chi < 0.05 else "Fail to reject H0: independent"
print(f"\nChi-square (international_plan vs churn): chi2={chi2:.3f}, p={p_chi:.6f}, dof={dof} -> {conclusion_chi}")
results.append(("chi-square: intl plan vs churn", chi2, p_chi, conclusion_chi))

# Second chi-square: voice mail plan vs churn
contingency2 = pd.crosstab(df["voice_mail_plan"], df["churn"])
chi2_2, p_chi2, dof2, _ = stats.chi2_contingency(contingency2)
conclusion_chi2 = "Reject H0: dependent" if p_chi2 < 0.05 else "Fail to reject H0: independent"
print(f"Chi-square (voice_mail_plan vs churn): chi2={chi2_2:.3f}, p={p_chi2:.6f} -> {conclusion_chi2}")
results.append(("chi-square: voicemail plan vs churn", chi2_2, p_chi2, conclusion_chi2))

# %% [markdown]
# ## 3. Confidence interval & margin of error for the overall churn rate
# 95% CI for a proportion using the normal approximation.

# %%
n = len(df)
p_hat = df["churn_flag"].mean()
z = stats.norm.ppf(0.975)  # 95% two-sided
margin_of_error = z * np.sqrt(p_hat * (1 - p_hat) / n)
ci_low, ci_high = p_hat - margin_of_error, p_hat + margin_of_error

print(f"\nChurn rate: {p_hat*100:.2f}%")
print(f"95% CI: [{ci_low*100:.2f}%, {ci_high*100:.2f}%]  (margin of error = {margin_of_error*100:.2f} pp)")
results.append(("95% CI churn rate (lower)", ci_low, np.nan, f"point est={p_hat:.4f}, MOE={margin_of_error:.4f}"))
results.append(("95% CI churn rate (upper)", ci_high, np.nan, ""))

# %% [markdown]
# ## 4. Probability distribution for risk analysis
# Model `customer_service_calls` as a Poisson process to estimate the
# probability a customer places > 4 calls (a known high-risk churn signal).

# %%
lam = df["customer_service_calls"].mean()
prob_more_than_4 = 1 - stats.poisson.cdf(4, mu=lam)
print(f"\nPoisson(lambda={lam:.2f}) fitted to service calls.")
print(f"P(service calls > 4) = {prob_more_than_4:.4f}")
results.append(("P(service_calls > 4) via Poisson", lam, prob_more_than_4, "risk probability estimate"))

# Empirical validation
empirical_prob = (df["customer_service_calls"] > 4).mean()
print(f"Empirical P(service calls > 4) = {empirical_prob:.4f}")

# %% [markdown]
# ## 5. A/B test simulation for a retention marketing campaign
# Simulates offering a retention discount (Group B) vs no offer (Group A) to
# customers flagged as high risk (>=4 service calls), and tests whether the
# campaign significantly reduces churn. Since we don't have real experiment
# data, we simulate a plausible outcome using the observed baseline churn
# rate and a assumed treatment lift, so you can see the exact mechanics
# you'd apply once real A/B data comes in — just replace the two arrays
# below with real group outcomes.

# %%
np.random.seed(42)
high_risk = df[df["customer_service_calls"] >= 4]
baseline_rate = high_risk["churn_flag"].mean() if len(high_risk) else df["churn_flag"].mean()

n_a, n_b = 150, 150
group_a = np.random.binomial(1, baseline_rate, n_a)              # control: no campaign
group_b = np.random.binomial(1, max(baseline_rate - 0.12, 0.02), n_b)  # treatment: assumed 12pp reduction from campaign

conv_a, conv_b = group_a.mean(), group_b.mean()
count = np.array([group_a.sum(), group_b.sum()])
nobs = np.array([n_a, n_b])
from statsmodels.stats.proportion import proportions_ztest
z_stat, p_ab = proportions_ztest(count, nobs)

conclusion_ab = "Campaign significantly reduces churn" if p_ab < 0.05 else "No significant difference detected"
print(f"\nA/B test — Group A (control) churn: {conv_a:.2%}, Group B (campaign) churn: {conv_b:.2%}")
print(f"z={z_stat:.3f}, p={p_ab:.5f} -> {conclusion_ab}")
results.append(("A/B test: retention campaign", z_stat, p_ab, conclusion_ab))

# %% [markdown]
# ## 6. Save all statistical results

# %%
results_df = pd.DataFrame(results, columns=["test", "statistic", "p_value", "conclusion"])
results_df.to_csv(os.path.join(PROCESSED_DIR, "statistical_test_results.csv"), index=False)
print("\nSaved statistical_test_results.csv")
print(results_df)
print("\nDone with Task: Statistical Analysis for Business Decisions.")
