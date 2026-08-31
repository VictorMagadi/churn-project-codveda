# =============================================================================
# Level 1 - Task 2: Data Collection and Cleaning (R version)
# Codveda Business Analytics Internship
# Run in RStudio. Uses only relative paths, so open this file in RStudio and
# either (a) set the project root as your working directory, or (b) just run
# the path-resolution block below, which finds the project root automatically
# when run via RStudio's "Source" 
# =============================================================================

# ---- 0. Setup -----------------------------------------------------------
required_pkgs <- c("tidyverse", "openxlsx")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(tidyverse)

# Resolve project root: works both when Sourced in RStudio and run via Rscript
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
raw_dir <- file.path(project_root, "data", "raw")
processed_dir <- file.path(project_root, "data", "processed")
dir.create(processed_dir, showWarnings = FALSE, recursive = TRUE)

cat("Project root:", project_root, "\n")

# ---- 1. Data Collection --------------------------------------------------
df_80 <- read_csv(file.path(raw_dir, "churn-bigml-80.csv"), show_col_types = FALSE) %>%
  mutate(source_split = "train_80")
df_20 <- read_csv(file.path(raw_dir, "churn-bigml-20.csv"), show_col_types = FALSE) %>%
  mutate(source_split = "holdout_20")

df_raw <- bind_rows(df_80, df_20)
cat(sprintf("Combined raw records: %d rows x %d columns\n", nrow(df_raw), ncol(df_raw)))

# ---- 2. Clean column names (snake_case) ----------------------------------
names(df_raw) <- names(df_raw) %>%
  str_trim() %>%
  str_to_lower() %>%
  str_replace_all(" ", "_") %>%
  str_replace_all("-", "_")

df <- df_raw

# ---- 3. Missing value profile ---------------------------------------------
missing_summary <- sapply(df, function(x) sum(is.na(x)))
print(missing_summary[missing_summary > 0])
cat("Total missing values:", sum(missing_summary), "\n")

# ---- 4. Impute missing values (median for numeric, mode for categorical) --
get_mode <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

numeric_cols <- names(df)[sapply(df, is.numeric)]
categorical_cols <- names(df)[sapply(df, function(x) is.character(x) || is.logical(x))]

for (col in numeric_cols) {
  if (any(is.na(df[[col]]))) {
    med <- median(df[[col]], na.rm = TRUE)
    df[[col]][is.na(df[[col]])] <- med
    cat(sprintf("Filled %s missing values with median=%.3f\n", col, med))
  }
}
for (col in categorical_cols) {
  if (any(is.na(df[[col]]))) {
    m <- get_mode(df[[col]])
    df[[col]][is.na(df[[col]])] <- m
    cat(sprintf("Filled %s missing values with mode=%s\n", col, m))
  }
}

# ---- 5. Drop duplicates + assign stable customer_id -----------------------
before_rows <- nrow(df)
df <- df %>% distinct()
cat(sprintf("Dropped %d duplicate rows. Remaining: %d\n", before_rows - nrow(df), nrow(df)))

df <- df %>%
  mutate(customer_id = sprintf("CUST%05d", row_number() - 1)) %>%
  relocate(customer_id)

# ---- 6. Outlier handling (IQR winsorization) -------------------------------
outlier_cols <- c("total_day_minutes", "total_day_charge", "total_eve_minutes",
                   "total_eve_charge", "total_night_minutes", "total_night_charge",
                   "total_intl_minutes", "total_intl_charge", "customer_service_calls")
outlier_cols <- intersect(outlier_cols, names(df))

outlier_report <- tibble()
for (col in outlier_cols) {
  q1 <- quantile(df[[col]], 0.25, na.rm = TRUE)
  q3 <- quantile(df[[col]], 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  n_out <- sum(df[[col]] < lower | df[[col]] > upper)
  outlier_report <- bind_rows(outlier_report, tibble(column = col, lower_fence = lower,
                                                       upper_fence = upper, n_outliers = n_out))
  df[[col]] <- pmin(pmax(df[[col]], lower), upper)  # winsorize/cap
}
print(outlier_report)

# ---- 7. Standardize categorical text ---------------------------------------
df <- df %>%
  mutate(
    international_plan = str_to_title(str_trim(international_plan)),
    voice_mail_plan = str_to_title(str_trim(voice_mail_plan)),
    state = str_to_upper(str_trim(state)),
    churn_flag = as.integer(churn)
  )

# ---- 8. Feature engineering --------------------------------------------
df <- df %>%
  mutate(
    total_minutes = total_day_minutes + total_eve_minutes + total_night_minutes + total_intl_minutes,
    total_charge = total_day_charge + total_eve_charge + total_night_charge + total_intl_charge,
    total_calls = total_day_calls + total_eve_calls + total_night_calls + total_intl_calls,
    avg_charge_per_call = ifelse(total_calls > 0, total_charge / total_calls, 0),
    has_intl_plan = as.integer(international_plan == "Yes"),
    has_voicemail_plan = as.integer(voice_mail_plan == "Yes")
  )

# ---- 9. Standardization (z-score) and normalization (min-max) ------------
scale_cols <- c("account_length", "total_day_minutes", "total_eve_minutes",
                 "total_night_minutes", "total_intl_minutes", "total_charge",
                 "customer_service_calls")
scale_cols <- intersect(scale_cols, names(df))

for (col in scale_cols) {
  df[[paste0(col, "_zscore")]] <- as.numeric(scale(df[[col]]))
  min_v <- min(df[[col]]); max_v <- max(df[[col]])
  df[[paste0(col, "_norm")]] <- if (max_v > min_v) (df[[col]] - min_v) / (max_v - min_v) else 0
}

# ---- 10. Train / test split (80/20, stratified by churn) -----------------
set.seed(42)
df <- df %>% mutate(row_id = row_number())
train_idx <- df %>% group_by(churn_flag) %>% slice_sample(prop = 0.8) %>% pull(row_id)
train_df <- df %>% filter(row_id %in% train_idx) %>% select(-row_id)
test_df <- df %>% filter(!(row_id %in% train_idx)) %>% select(-row_id)
df <- df %>% select(-row_id)

cat(sprintf("Train: %d rows, Test: %d rows\n", nrow(train_df), nrow(test_df)))

# ---- 11. Save outputs ------------------------------------------------------
write_csv(df, file.path(processed_dir, "churn_cleaned_R.csv"))
write_csv(train_df, file.path(processed_dir, "churn_train_R.csv"))
write_csv(test_df, file.path(processed_dir, "churn_test_R.csv"))
write_csv(outlier_report, file.path(processed_dir, "outlier_report_R.csv"))

cat("\nSaved churn_cleaned_R.csv, churn_train_R.csv, churn_test_R.csv, outlier_report_R.csv\n")
cat("Done with Task: Data Collection & Cleaning (R).\n")
