# =============================================================================
# Level 2 - Task 3: Statistical Analysis for Business Decisions (R version)
# Codveda Business Analytics Internship
# Run 01_data_cleaning.R first.
# =============================================================================

library(tidyverse)

get_script_dir <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    return(dirname(rstudioapi::getSourceEditorContext()$path))
  }
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("--file=", "", args[grep("--file=", args)])
  if (length(file_arg)) return(dirname(normalizePath(file_arg)))
  return(getwd())
}
script_dir <- get_script_dir()
project_root <- normalizePath(file.path(script_dir, ".."))
processed_dir <- file.path(project_root, "data", "processed")

df <- read_csv(file.path(processed_dir, "churn_cleaned_R.csv"), show_col_types = FALSE)
results <- tibble(test = character(), statistic = double(), p_value = double(), conclusion = character())

# ---- 1. T-tests -------------------------------------------------------------
t1 <- t.test(total_day_minutes ~ churn, data = df)
cat(sprintf("T-test (day minutes, churn vs stay): t=%.3f, p=%.5f\n", t1$statistic, t1$p.value))
results <- add_row(results, test = "t-test: day minutes churn vs stay",
                    statistic = as.numeric(t1$statistic), p_value = t1$p.value,
                    conclusion = ifelse(t1$p.value < 0.05, "Reject H0: means differ", "Fail to reject H0"))

t2 <- t.test(customer_service_calls ~ churn, data = df)
cat(sprintf("T-test (service calls, churn vs stay): t=%.3f, p=%.5f\n", t2$statistic, t2$p.value))
results <- add_row(results, test = "t-test: service calls churn vs stay",
                    statistic = as.numeric(t2$statistic), p_value = t2$p.value,
                    conclusion = ifelse(t2$p.value < 0.05, "Reject H0: means differ", "Fail to reject H0"))

# ---- 2. Chi-square tests -----------------------------------------------------
tbl1 <- table(df$international_plan, df$churn)
chi1 <- chisq.test(tbl1)
cat(sprintf("\nChi-square (intl plan vs churn): chi2=%.3f, p=%.6f\n", chi1$statistic, chi1$p.value))
results <- add_row(results, test = "chi-square: intl plan vs churn",
                    statistic = as.numeric(chi1$statistic), p_value = chi1$p.value,
                    conclusion = ifelse(chi1$p.value < 0.05, "Reject H0: dependent", "Fail to reject H0"))

tbl2 <- table(df$voice_mail_plan, df$churn)
chi2 <- chisq.test(tbl2)
cat(sprintf("Chi-square (voicemail plan vs churn): chi2=%.3f, p=%.6f\n", chi2$statistic, chi2$p.value))
results <- add_row(results, test = "chi-square: voicemail plan vs churn",
                    statistic = as.numeric(chi2$statistic), p_value = chi2$p.value,
                    conclusion = ifelse(chi2$p.value < 0.05, "Reject H0: dependent", "Fail to reject H0"))

# ---- 3. Confidence interval for churn rate ----------------------------------
n <- nrow(df)
p_hat <- mean(df$churn_flag)
z <- qnorm(0.975)
moe <- z * sqrt(p_hat * (1 - p_hat) / n)
cat(sprintf("\nChurn rate: %.2f%%, 95%% CI: [%.2f%%, %.2f%%], MOE=%.2f pp\n",
            p_hat * 100, (p_hat - moe) * 100, (p_hat + moe) * 100, moe * 100))
results <- add_row(results, test = "95% CI churn rate", statistic = p_hat, p_value = NA,
                    conclusion = sprintf("MOE=%.4f, CI=[%.4f, %.4f]", moe, p_hat - moe, p_hat + moe))

# ---- 4. Probability distribution (Poisson) for risk -------------------------
lambda <- mean(df$customer_service_calls)
prob_gt4 <- 1 - ppois(4, lambda)
cat(sprintf("\nPoisson(lambda=%.2f). P(service calls > 4) = %.4f\n", lambda, prob_gt4))
results <- add_row(results, test = "P(service_calls > 4) via Poisson", statistic = lambda,
                    p_value = prob_gt4, conclusion = "risk probability estimate")

# ---- 5. A/B test simulation for a retention campaign ------------------------
set.seed(42)
high_risk <- df %>% filter(customer_service_calls >= 4)
baseline_rate <- if (nrow(high_risk) > 0) mean(high_risk$churn_flag) else mean(df$churn_flag)

n_a <- 150; n_b <- 150
group_a <- rbinom(n_a, 1, baseline_rate)
group_b <- rbinom(n_b, 1, max(baseline_rate - 0.12, 0.02))

prop_test <- prop.test(c(sum(group_a), sum(group_b)), c(n_a, n_b))
cat(sprintf("\nA/B test - Control churn: %.2f%%, Campaign churn: %.2f%%, p=%.5f\n",
            mean(group_a) * 100, mean(group_b) * 100, prop_test$p.value))
results <- add_row(results, test = "A/B test: retention campaign",
                    statistic = as.numeric(prop_test$statistic), p_value = prop_test$p.value,
                    conclusion = ifelse(prop_test$p.value < 0.05,
                                         "Campaign significantly reduces churn", "No significant difference"))

# ---- 6. Save results ----------------------------------------------------------
write_csv(results, file.path(processed_dir, "statistical_test_results_R.csv"))
print(results)
cat("\nDone with Task: Statistical Analysis for Business Decisions (R).\n")
