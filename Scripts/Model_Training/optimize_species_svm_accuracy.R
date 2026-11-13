# ============================================================
# SPECIES SVM OPTIMIZATION - PRIORITIZE ACCURACY
# Uses the existing common 80:20 split and 33 audio features
# ============================================================

if (!requireNamespace("e1071", quietly = TRUE)) {
  install.packages("e1071")
}
library(e1071)

# 1. Paths
split_dir <- "Data_Processed/Common_Splits"
output_dir <- "Data_Processed/Species_SVM_Accuracy_Optimization"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train_file <- file.path(split_dir, "Species_train.csv")
test_file  <- file.path(split_dir, "Species_test.csv")

if (!file.exists(train_file) || !file.exists(test_file)) {
  stop("Species common-split files were not found. Run this script from the DA-Project root.")
}

# 2. Load data
train_data <- read.csv(train_file, check.names = FALSE)
test_data  <- read.csv(test_file, check.names = FALSE)

# 3. Select the same 33 audio features
features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)

target <- "Species"
required <- c("File_Name", target, features)

if (!all(required %in% names(train_data)) ||
    !all(required %in% names(test_data))) {
  stop("One or more required columns are missing from the Species split files.")
}

# Keep rows with known target labels
train_data <- train_data[!is.na(train_data[[target]]) &
                           trimws(as.character(train_data[[target]])) != "", ]
test_data <- test_data[!is.na(test_data[[target]]) &
                         trimws(as.character(test_data[[target]])) != "", ]

# 4. Prepare class labels
class_levels <- sort(unique(as.character(train_data[[target]])))

if (any(!as.character(test_data[[target]]) %in% class_levels)) {
  stop("Test data contains a species class that is absent from training data.")
}

y_train <- factor(as.character(train_data[[target]]), levels = class_levels)
y_test  <- factor(as.character(test_data[[target]]), levels = class_levels)

# Each species needs at least 5 training rows for stratified 5-fold CV
class_counts <- table(y_train)
if (any(class_counts < 5)) {
  stop("At least one species has fewer than 5 training records; 5-fold stratified CV is not suitable.")
}

# 5. Convert feature columns to numeric and validate
x_train <- train_data[, features, drop = FALSE]
x_test  <- test_data[, features, drop = FALSE]

x_train[] <- lapply(x_train, function(x) suppressWarnings(as.numeric(as.character(x))))
x_test[]  <- lapply(x_test, function(x) suppressWarnings(as.numeric(as.character(x))))

if (any(!is.finite(as.matrix(x_train))) || any(!is.finite(as.matrix(x_test)))) {
  stop("Audio features contain missing or non-finite values. Resolve these before running SVM tuning.")
}

# 6. Stratified folds (same folds used for every configuration)
make_stratified_folds <- function(y, k = 5, seed = 2026) {
  set.seed(seed)
  fold_id <- integer(length(y))
  for (class_name in levels(y)) {
    idx <- which(y == class_name)
    idx <- sample(idx)
    fold_id[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
  }
  fold_id
}

fold_id <- make_stratified_folds(y_train, k = 5, seed = 2026)

# 7. Class weights: compare no weighting with two weighting strengths
get_class_weights <- function(y, method, class_levels) {
  if (method == "none") return(NULL)

  counts <- table(factor(y, levels = class_levels))
  if (any(counts == 0)) stop("A CV training fold is missing a species class.")

  weights <- length(y) / (length(class_levels) * as.numeric(counts))
  if (method == "sqrt_inverse") weights <- sqrt(weights)
  names(weights) <- class_levels
  weights
}

# 8. Metrics from a confusion matrix
calculate_metrics <- function(actual, predicted, class_levels) {
  cm <- table(
    Actual = factor(actual, levels = class_levels),
    Predicted = factor(predicted, levels = class_levels)
  )

  tp <- diag(cm)
  precision <- tp / pmax(colSums(cm), 1)
  recall <- tp / pmax(rowSums(cm), 1)
  f1 <- ifelse(precision + recall == 0, 0,
               2 * precision * recall / (precision + recall))

  list(
    accuracy = sum(tp) / sum(cm),
    balanced_accuracy = mean(recall),
    macro_f1 = mean(f1),
    confusion = cm,
    class_metrics = data.frame(
      Species = class_levels,
      Support = as.integer(rowSums(cm)),
      Precision = as.numeric(precision),
      Recall = as.numeric(recall),
      F1_Score = as.numeric(f1)
    )
  )
}

# 9. Hyperparameter grid
# 3 weighting choices x 5 cost values x 5 gamma values = 75 models
parameter_grid <- expand.grid(
  Weighting = c("none", "sqrt_inverse", "inverse"),
  Cost = c(0.1, 1, 10, 100, 1000),
  Gamma = c(0.0001, 0.001, 0.01, 0.1, 1),
  stringsAsFactors = FALSE
)

cat("Training records:", nrow(x_train), "\n")
cat("Testing records:", nrow(x_test), "\n")
cat("Species classes:", length(class_levels), "\n")
cat("Parameter combinations:", nrow(parameter_grid), "\n")
cat("Starting stratified 5-fold CV...\n")

# 10. Cross-validation: choose using CV accuracy, not test accuracy
cv_results <- vector("list", nrow(parameter_grid))

for (i in seq_len(nrow(parameter_grid))) {
  cfg <- parameter_grid[i, ]
  oof_predictions <- rep(NA_character_, length(y_train))

  for (fold in 1:5) {
    valid_idx <- which(fold_id == fold)
    train_idx <- which(fold_id != fold)

    y_fold_train <- droplevels(y_train[train_idx])
    weights <- get_class_weights(y_fold_train, cfg$Weighting, class_levels)

    model <- e1071::svm(
      x = x_train[train_idx, , drop = FALSE],
      y = y_fold_train,
      type = "C-classification",
      kernel = "radial",
      cost = cfg$Cost,
      gamma = cfg$Gamma,
      class.weights = weights,
      scale = TRUE
    )

    pred <- predict(model, newdata = x_train[valid_idx, , drop = FALSE])
    oof_predictions[valid_idx] <- as.character(pred)
  }

  m <- calculate_metrics(y_train, oof_predictions, class_levels)

  cv_results[[i]] <- data.frame(
    Weighting = cfg$Weighting,
    Cost = cfg$Cost,
    Gamma = cfg$Gamma,
    CV_Accuracy = m$accuracy,
    CV_Balanced_Accuracy = m$balanced_accuracy,
    CV_Macro_F1 = m$macro_f1
  )

  cat(sprintf(
    "Completed %d/%d | Weight=%s | C=%g | Gamma=%g | CV Accuracy=%.4f | Macro-F1=%.4f\n",
    i, nrow(parameter_grid), cfg$Weighting, cfg$Cost, cfg$Gamma,
    m$accuracy, m$macro_f1
  ))
}

cv_results <- do.call(rbind, cv_results)
cv_results <- cv_results[order(
  -cv_results$CV_Accuracy,
  -cv_results$CV_Macro_F1,
  -cv_results$CV_Balanced_Accuracy
), ]

best <- cv_results[1, ]
cat("\nBest configuration selected by CV Accuracy:\n")
print(best)

# 11. Fit selected configuration on all training data
final_weights <- get_class_weights(y_train, best$Weighting, class_levels)

final_model <- e1071::svm(
  x = x_train,
  y = y_train,
  type = "C-classification",
  kernel = "radial",
  cost = best$Cost,
  gamma = best$Gamma,
  class.weights = final_weights,
  scale = TRUE
)

# 12. Evaluate once on the held-out test split
predictions <- predict(final_model, newdata = x_test)
metrics <- calculate_metrics(y_test, predictions, class_levels)

cat("\n========== FINAL SPECIES SVM TEST RESULTS ==========\n")
cat("Weighting:", best$Weighting, "\n")
cat("C:", best$Cost, "\n")
cat("Gamma:", best$Gamma, "\n")
cat(sprintf("Accuracy: %.2f%%\n", metrics$accuracy * 100))
cat(sprintf("Balanced Accuracy: %.2f%%\n", metrics$balanced_accuracy * 100))
cat(sprintf("Macro-F1: %.2f%%\n", metrics$macro_f1 * 100))
cat("\nConfusion Matrix (rows=actual, columns=predicted):\n")
print(metrics$confusion)
cat("\nClass-wise Metrics:\n")
print(metrics$class_metrics)

# 13. Save outputs
write.csv(cv_results,
          file.path(output_dir, "species_svm_cv_results.csv"),
          row.names = FALSE)
write.csv(as.data.frame.matrix(metrics$confusion),
          file.path(output_dir, "species_svm_confusion_matrix.csv"))
write.csv(metrics$class_metrics,
          file.path(output_dir, "species_svm_class_metrics.csv"),
          row.names = FALSE)
write.csv(data.frame(
  File_Name = test_data$File_Name,
  Actual = y_test,
  Predicted = predictions
), file.path(output_dir, "species_svm_test_predictions.csv"),
row.names = FALSE)

saveRDS(list(
  model = final_model,
  features = features,
  class_levels = class_levels,
  best_parameters = best
), file.path(output_dir, "species_svm_best_model.rds"))

write.csv(data.frame(
  Task = "Species",
  Weighting = best$Weighting,
  Cost = best$Cost,
  Gamma = best$Gamma,
  CV_Accuracy = best$CV_Accuracy,
  Test_Accuracy = metrics$accuracy,
  Test_Balanced_Accuracy = metrics$balanced_accuracy,
  Test_Macro_F1 = metrics$macro_f1
), file.path(output_dir, "species_svm_final_summary.csv"),
row.names = FALSE)

cat("\nResults saved in:", output_dir, "\n")
