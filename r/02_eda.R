# =============================================================================
# Level 1 - Task 3: Exploratory Data Analysis (R version)
# Codveda Business Analytics Internship
# Run 01_data_cleaning.R first.
# =============================================================================

required_pkgs <- c("tidyverse", "corrplot")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(tidyverse)
library(corrplot)

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
dir.create(visuals_dir, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(file.path(processed_dir, "churn_cleaned_R.csv"), show_col_types = FALSE)

# ---- 1. Statistical summaries ---------------------------------------------
numeric_cols <- names(df)[sapply(df, is.numeric)]
summary_stats <- df %>%
  summarise(across(all_of(numeric_cols), list(mean = ~mean(.x), median = ~median(.x),
                                               sd = ~sd(.x), min = ~min(.x), max = ~max(.x)),
                    .names = "{.col}__{.fn}")) %>%
  pivot_longer(everything(), names_to = c("column", "stat"), names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value)
print(summary_stats)
write_csv(summary_stats, file.path(processed_dir, "eda_summary_stats_R.csv"))

# ---- 2. Churn rate overview -------------------------------------------------
churn_rate <- mean(df$churn_flag) * 100
cat(sprintf("Overall churn rate: %.2f%%\n", churn_rate))

churn_by_state <- df %>% group_by(state) %>% summarise(churn_rate_pct = mean(churn_flag) * 100) %>%
  arrange(desc(churn_rate_pct))
print(head(churn_by_state, 10))

# ---- 3. Churn distribution plot --------------------------------------------
p1 <- ggplot(df, aes(x = churn, fill = churn)) +
  geom_bar() +
  scale_fill_manual(values = c("FALSE" = "#2E86AB", "TRUE" = "#E63946")) +
  labs(title = "Customer Churn Distribution", x = "Churned", y = "Count") +
  theme_minimal()
ggsave(file.path(visuals_dir, "R_01_churn_distribution.png"), p1, width = 5, height = 4, dpi = 150)

# ---- 4. Histograms of key usage metrics ------------------------------------
hist_cols <- c("total_day_minutes", "total_eve_minutes", "total_night_minutes",
               "total_intl_minutes", "customer_service_calls", "account_length")
df_long <- df %>% select(all_of(hist_cols)) %>% pivot_longer(everything())
p2 <- ggplot(df_long, aes(x = value)) +
  geom_histogram(bins = 30, fill = "#2E86AB", color = "white") +
  facet_wrap(~name, scales = "free") +
  labs(title = "Distributions of Key Usage Metrics") +
  theme_minimal()
ggsave(file.path(visuals_dir, "R_02_histograms.png"), p2, width = 12, height = 7, dpi = 150)

# ---- 5. Scatter plots -------------------------------------------------------
p3 <- ggplot(df, aes(x = total_day_minutes, y = total_day_charge, color = churn)) +
  geom_point(alpha = 0.4) +
  scale_color_manual(values = c("FALSE" = "#2E86AB", "TRUE" = "#E63946")) +
  labs(title = "Total Day Minutes vs Charge") +
  theme_minimal()
ggsave(file.path(visuals_dir, "R_03_scatter_minutes_vs_charge.png"), p3, width = 7, height = 5, dpi = 150)

# ---- 6. Correlation matrix --------------------------------------------------
corr_cols <- c("account_length", "total_day_minutes", "total_eve_minutes", "total_night_minutes",
               "total_intl_minutes", "customer_service_calls", "total_charge", "churn_flag")
corr_matrix <- cor(df %>% select(all_of(corr_cols)))
print(sort(corr_matrix[, "churn_flag"], decreasing = TRUE))

png(file.path(visuals_dir, "R_04_correlation_heatmap.png"), width = 900, height = 800, res = 150)
corrplot(corr_matrix, method = "color", type = "upper", addCoef.col = "black",
         number.cex = 0.6, tl.col = "black", tl.cex = 0.8,
         title = "Correlation Matrix - Usage vs Churn", mar = c(0, 0, 2, 0))
dev.off()

# ---- 7. Churn rate by international plan & service calls -------------------
p4a <- df %>% group_by(international_plan) %>% summarise(churn_pct = mean(churn_flag) * 100) %>%
  ggplot(aes(x = international_plan, y = churn_pct, fill = international_plan)) +
  geom_col() + labs(title = "Churn Rate by International Plan", y = "Churn %") + theme_minimal()
ggsave(file.path(visuals_dir, "R_05a_churn_by_intl_plan.png"), p4a, width = 6, height = 4.5, dpi = 150)

p4b <- df %>% group_by(customer_service_calls) %>% summarise(churn_pct = mean(churn_flag) * 100) %>%
  ggplot(aes(x = customer_service_calls, y = churn_pct)) +
  geom_col(fill = "#E63946") + labs(title = "Churn Rate by # Service Calls", y = "Churn %") + theme_minimal()
ggsave(file.path(visuals_dir, "R_05b_churn_by_service_calls.png"), p4b, width = 6, height = 4.5, dpi = 150)

cat("\nAll R EDA visuals saved to:", visuals_dir, "\n")
cat("Done with Task: Exploratory Data Analysis (R).\n")
