# =============================================================================
# Level 3 - Task 1: Predictive Analytics & Machine Learning (R version)
# Codveda Business Analytics Internship
# Run 01_data_cleaning.R first.
# Covers: regression, classification (churn), and clustering (segmentation).
# =============================================================================

required_pkgs <- c("tidyverse", "caret", "randomForest", "pROC", "cluster", "factoextra")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs)) install.packages(new_pkgs)

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(cluster)
library(factoextra)

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

train_df <- read_csv(file.path(processed_dir, "churn_train_R.csv"), show_col_types = FALSE)
test_df <- read_csv(file.path(processed_dir, "churn_test_R.csv"), show_col_types = FALSE)

feature_cols <- c("account_length", "total_day_minutes", "total_day_calls",
                   "total_eve_minutes", "total_eve_calls", "total_night_minutes",
                   "total_night_calls", "total_intl_minutes", "total_intl_calls",
                   "customer_service_calls", "has_intl_plan", "has_voicemail_plan",
                   "number_vmail_messages")
feature_cols <- intersect(feature_cols, names(train_df))

# ---- 1. Regression: forecast total_charge -----------------------------------
lm_formula <- as.formula(paste("total_charge ~", paste(feature_cols, collapse = " + ")))
lm_model <- lm(lm_formula, data = train_df)
pred_lm <- predict(lm_model, newdata = test_df)
rmse_lm <- sqrt(mean((test_df$total_charge - pred_lm)^2))
r2_lm <- cor(test_df$total_charge, pred_lm)^2
cat(sprintf("[Linear Regression] RMSE=%.3f, R2=%.4f\n", rmse_lm, r2_lm))

rf_reg_formula <- lm_formula
rf_reg <- randomForest(rf_reg_formula, data = train_df, ntree = 300)
pred_rf_reg <- predict(rf_reg, newdata = test_df)
rmse_rf <- sqrt(mean((test_df$total_charge - pred_rf_reg)^2))
r2_rf <- cor(test_df$total_charge, pred_rf_reg)^2
cat(sprintf("[Random Forest Regressor] RMSE=%.3f, R2=%.4f\n", rmse_rf, r2_rf))

# ---- 2. Classification: churn prediction -------------------------------------
train_df$churn_factor <- factor(train_df$churn_flag, levels = c(0, 1), labels = c("Stayed", "Churned"))
test_df$churn_factor <- factor(test_df$churn_flag, levels = c(0, 1), labels = c("Stayed", "Churned"))

glm_formula <- as.formula(paste("churn_factor ~", paste(feature_cols, collapse = " + ")))
log_model <- glm(as.formula(paste("churn_flag ~", paste(feature_cols, collapse = " + "))),
                  data = train_df, family = "binomial")
prob_log <- predict(log_model, newdata = test_df, type = "response")
pred_log <- factor(ifelse(prob_log > 0.5, "Churned", "Stayed"), levels = c("Stayed", "Churned"))

rf_clf <- randomForest(glm_formula, data = train_df, ntree = 400, importance = TRUE)
pred_rf_clf <- predict(rf_clf, newdata = test_df)
prob_rf_clf <- predict(rf_clf, newdata = test_df, type = "prob")[, "Churned"]

cm_log <- confusionMatrix(pred_log, test_df$churn_factor, positive = "Churned")
cm_rf <- confusionMatrix(pred_rf_clf, test_df$churn_factor, positive = "Churned")
cat("\n--- Logistic Regression ---\n"); print(cm_log$overall[c("Accuracy")]); print(cm_log$byClass[c("Precision", "Recall", "F1")])
cat("\n--- Random Forest Classifier ---\n"); print(cm_rf$overall[c("Accuracy")]); print(cm_rf$byClass[c("Precision", "Recall", "F1")])

roc_log <- roc(test_df$churn_flag, prob_log, quiet = TRUE)
roc_rf <- roc(test_df$churn_flag, prob_rf_clf, quiet = TRUE)
cat(sprintf("\nROC-AUC Logistic: %.4f | ROC-AUC RandomForest: %.4f\n", auc(roc_log), auc(roc_rf)))

png(file.path(visuals_dir, "R_08_roc_curves.png"), width = 800, height = 700, res = 150)
plot(roc_rf, col = "#E63946", main = "ROC Curves - Churn Classifiers (R)")
lines(roc_log, col = "#2E86AB")
legend("bottomright", legend = c(sprintf("RF (AUC=%.3f)", auc(roc_rf)),
                                  sprintf("LogReg (AUC=%.3f)", auc(roc_log))),
       col = c("#E63946", "#2E86AB"), lwd = 2)
dev.off()

# Feature importance
png(file.path(visuals_dir, "R_09_feature_importance.png"), width = 900, height = 700, res = 150)
varImpPlot(rf_clf, main = "Feature Importance - Churn Prediction (R RandomForest)")
dev.off()

# ---- 3. Clustering: customer segmentation (KMeans) ---------------------------
cluster_features <- c("total_day_minutes", "total_eve_minutes", "total_night_minutes",
                       "total_intl_minutes", "customer_service_calls", "account_length")
full_df <- bind_rows(train_df, test_df)
X_cluster <- scale(full_df[, cluster_features])

set.seed(42)
fviz_nbclust(X_cluster, kmeans, method = "silhouette", k.max = 7) -> sil_plot
ggsave(file.path(visuals_dir, "R_10_clustering_silhouette.png"), sil_plot, width = 7, height = 5, dpi = 150)

km_result <- kmeans(X_cluster, centers = 2, nstart = 25)
full_df$customer_segment <- km_result$cluster

segment_profile <- full_df %>%
  group_by(customer_segment) %>%
  summarise(across(all_of(c(cluster_features, "churn_flag")), mean))
print(segment_profile)
write_csv(segment_profile, file.path(processed_dir, "customer_segment_profiles_R.csv"))

# ---- 4. Save scored predictions ------------------------------------------------
full_df$predicted_churn_probability <- predict(rf_clf, newdata = full_df, type = "prob")[, "Churned"]
full_df$predicted_churn_risk_tier <- cut(full_df$predicted_churn_probability,
                                          breaks = c(-0.01, 0.3, 0.6, 1.0),
                                          labels = c("Low", "Medium", "High"))

out <- full_df %>% select(customer_id, state, account_length, churn, churn_flag,
                           customer_segment, predicted_churn_probability, predicted_churn_risk_tier)
write_csv(out, file.path(processed_dir, "model_predictions_R.csv"))

cat("\nSaved model_predictions_R.csv, customer_segment_profiles_R.csv\n")
cat("Done with Task: Predictive Analytics & Machine Learning (R).\n")
