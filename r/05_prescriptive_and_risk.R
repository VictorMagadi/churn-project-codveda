# =============================================================================
# Level 3 - Task 2 & 3: Risk/Anomaly Detection + Prescriptive Analytics (R)
# Codveda Business Analytics Internship
# Run 01_data_cleaning.R and 04_predictive_modeling.R first.
# =============================================================================

required_pkgs <- c("tidyverse", "solitude", "lpSolve")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(tidyverse)
library(lpSolve)
# `solitude` provides an R implementation of Isolation Forest. If it's not
# available in your environment, this script falls back automatically to a
# robust Mahalanobis-distance anomaly score (also a standard multivariate
# outlier detection method).
has_solitude <- requireNamespace("solitude", quietly = TRUE)

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
visuals_dir <- file.path(project_root, "visuals")

df <- read_csv(file.path(processed_dir, "churn_cleaned_R.csv"), show_col_types = FALSE)
predictions <- read_csv(file.path(processed_dir, "model_predictions_R.csv"), show_col_types = FALSE)

# =========================== PART A: RISK & ANOMALY DETECTION ===============

anomaly_features <- c("total_day_minutes", "total_day_calls", "total_day_charge",
                       "total_eve_minutes", "total_eve_calls", "total_night_minutes",
                       "total_night_calls", "total_intl_minutes", "total_intl_calls",
                       "total_intl_charge", "customer_service_calls")
X <- scale(df[, anomaly_features])

if (has_solitude) {
  library(solitude)
  iso <- isolationForest$new(sample_size = 256, num_trees = 300, seed = 42)
  iso$fit(as.data.frame(X))
  scores <- iso$predict(as.data.frame(X))
  df$anomaly_score <- scores$anomaly_score
} else {
  # Fallback: Mahalanobis distance as multivariate anomaly score
  center <- colMeans(X)
  cov_mat <- cov(X)
  df$anomaly_score <- mahalanobis(X, center, cov_mat)
  df$anomaly_score <- (df$anomaly_score - min(df$anomaly_score)) /
    (max(df$anomaly_score) - min(df$anomaly_score))
}

anomaly_cutoff <- quantile(df$anomaly_score, 0.97)
df$anomaly_flag <- as.integer(df$anomaly_score >= anomaly_cutoff)
cat(sprintf("Flagged %d anomalous records (top 3%% by anomaly score)\n", sum(df$anomaly_flag)))

# Rule-based risk indicators
df <- df %>%
  mutate(
    rule_excessive_intl_usage = as.integer(total_intl_calls > quantile(total_intl_calls, 0.98)),
    rule_excessive_service_calls = as.integer(customer_service_calls >= 6),
    rule_unusual_charge_ratio = as.integer(avg_charge_per_call > quantile(avg_charge_per_call, 0.98)),
  ) %>%
  mutate(risk_rule_score = rule_excessive_intl_usage + rule_excessive_service_calls + rule_unusual_charge_ratio)

df <- df %>% left_join(predictions %>% select(customer_id, predicted_churn_probability), by = "customer_id")

minmax <- function(x) if (max(x) > min(x)) (x - min(x)) / (max(x) - min(x)) else x * 0
df <- df %>%
  mutate(
    predicted_churn_probability = ifelse(is.na(predicted_churn_probability),
                                          mean(predicted_churn_probability, na.rm = TRUE),
                                          predicted_churn_probability),
    risk_score = 100 * (0.4 * minmax(anomaly_score) + 0.3 * minmax(risk_rule_score) + 0.3 * predicted_churn_probability),
    risk_tier = cut(risk_score, breaks = c(-1, 33, 66, 100), labels = c("Low", "Medium", "High"))
  )

print(table(df$risk_tier))

p_risk <- ggplot(df, aes(x = risk_score)) +
  geom_histogram(bins = 30, fill = "#E63946", color = "white") +
  labs(title = "Distribution of Composite Risk Scores (R)") +
  theme_minimal()
ggsave(file.path(visuals_dir, "R_11_risk_score_distribution.png"), p_risk, width = 8, height = 5, dpi = 150)

risk_out <- df %>% select(customer_id, state, account_length, churn, customer_service_calls,
                           total_intl_calls, anomaly_score, anomaly_flag, risk_rule_score,
                           predicted_churn_probability, risk_score, risk_tier)
write_csv(risk_out, file.path(processed_dir, "customer_risk_scores_R.csv"))
cat("Saved customer_risk_scores_R.csv\n")

# =========================== PART B: PRESCRIPTIVE ANALYTICS =================
# Linear programming: allocate retention outreach across risk tiers to
# maximize expected retained annual value, subject to budget & staffing caps.

tier_counts <- table(risk_out$risk_tier)
tiers <- c("High", "Medium", "Low")
tier_counts <- tier_counts[tiers]

avg_customer_monthly_value <- 60
retention_uplift <- c(High = 0.35, Medium = 0.18, Low = 0.05)
cost_per_outreach <- c(High = 15, Medium = 8, Low = 3)
hours_per_outreach <- c(High = 0.5, Medium = 0.25, Low = 0.1)
TOTAL_BUDGET <- 5000
MAX_STAFF_HOURS <- 400

objective <- retention_uplift[tiers] * avg_customer_monthly_value * 12  # maximize
const_mat <- rbind(cost_per_outreach[tiers], hours_per_outreach[tiers])
const_dir <- c("<=", "<=")
const_rhs <- c(TOTAL_BUDGET, MAX_STAFF_HOURS)

lp_result <- lp("max", objective, const_mat, const_dir, const_rhs,
                 all.int = FALSE, upper = as.numeric(tier_counts))

allocation <- setNames(round(lp_result$solution), tiers)
cat("\nOptimal outreach allocation:\n"); print(allocation)
cat(sprintf("Expected annualized retained value: $%.0f\n", lp_result$objval))

allocation_df <- tibble(
  risk_tier = tiers,
  customers_to_contact = as.integer(allocation),
  cost_per_customer = cost_per_outreach[tiers],
  total_cost = as.integer(allocation) * cost_per_outreach[tiers],
  retention_uplift_pct = retention_uplift[tiers] * 100
)
write_csv(allocation_df, file.path(processed_dir, "optimal_retention_allocation_R.csv"))

# Scenario analysis across budgets
scenario_results <- map_dfr(c(2000, 5000, 8000, 12000, 20000), function(budget) {
  res <- lp("max", objective, const_mat, const_dir, c(budget, MAX_STAFF_HOURS),
            all.int = FALSE, upper = as.numeric(tier_counts))
  tibble(budget = budget, expected_retained_value = res$objval)
})
print(scenario_results)
write_csv(scenario_results, file.path(processed_dir, "budget_scenario_analysis_R.csv"))

p_scenario <- ggplot(scenario_results, aes(x = budget, y = expected_retained_value)) +
  geom_line(color = "#2E86AB") + geom_point(color = "#2E86AB", size = 2) +
  labs(title = "Scenario Analysis: Budget vs Expected Retained Value (R)",
       x = "Retention budget ($)", y = "Expected annualized value ($)") +
  theme_minimal()
ggsave(file.path(visuals_dir, "R_13_budget_scenario_analysis.png"), p_scenario, width = 8, height = 5, dpi = 150)

cat("\nSaved optimal_retention_allocation_R.csv, budget_scenario_analysis_R.csv\n")
cat("Done with Task: Risk/Anomaly Detection & Prescriptive Analytics (R).\n")
