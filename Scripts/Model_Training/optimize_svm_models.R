
# ============================================================
# SVM HYPERPARAMETER OPTIMIZATION
# Species Classification and Threat Classification
# Optimization Metric: Macro F1
# ============================================================

# Install and load required package
if (!requireNamespace("e1071", quietly = TRUE)) {
  install.packages("e1071")
}

library(e1071)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT <- getwd()

SPLIT_DIR <- file.path(
  PROJECT_ROOT, "Data_Processed", "Common_Splits"
)

MODEL_DIR <- file.path(
  PROJECT_ROOT, "Data_Processed", "Models"
)

PREDICTION_DIR <- file.path(
  PROJECT_ROOT, "Data_Processed", "Predictions"
)

EVALUATION_DIR <- file.path(
  PROJECT_ROOT, "Data_Processed", "Evaluation"
)

dir.create(MODEL_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(PREDICTION_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(EVALUATION_DIR, showWarnings = FALSE, recursive = TRUE)


# ============================================================
# 2. SELECT 33 AUDIO FEATURES
# ============================================================

feature_cols <- c(
  "Pitch_Centroid",
  "Energy",
  "ZCR",
  "Spectral_Bandwidth",
  "Spectral_RollOff85",
  "Spectral_Flatness",
  "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)


# ============================================================
# 3. CREATE STRATIFIED 5-FOLD CROSS VALIDATION
# ============================================================

create_folds <- function(y, k = 5) {

  fold_id <- integer(length(y))

  for (class_name in levels(y)) {

    # Find records of the current class
    indices <- which(y == class_name)

    # Shuffle records
    indices <- sample(indices)

    # Distribute records across folds
    fold_id[indices] <- sample(
      rep(1:k, length.out = length(indices))
    )
  }

  return(fold_id)
}


# ============================================================
# 4. CALCULATE CLASS WEIGHTS
# ============================================================

calculate_class_weights <- function(y, class_levels) {

  # Count records in each class
  counts <- table(factor(y, levels = class_levels))

  # Give minority classes higher weights
  weights <- length(y) / (
    length(class_levels) * as.numeric(counts)
  )

  names(weights) <- class_levels

  return(weights)
}


# ============================================================
# 5. CALCULATE EVALUATION METRICS
# ============================================================

calculate_metrics <- function(actual, predicted, class_levels) {

  actual <- factor(actual, levels = class_levels)

  predicted <- factor(predicted, levels = class_levels)

  # Create confusion matrix
  cm <- table(
    Actual = actual,
    Predicted = predicted
  )

  correct <- diag(cm)

  # Calculate precision
  precision <- ifelse(
    colSums(cm) == 0,
    0,
    correct / colSums(cm)
  )

  # Calculate recall
  recall <- ifelse(
    rowSums(cm) == 0,
    0,
    correct / rowSums(cm)
  )

  # Calculate F1 score
  f1 <- ifelse(
    precision + recall == 0,
    0,
    2 * precision * recall / (precision + recall)
  )

  # Overall accuracy
  accuracy <- sum(correct) / sum(cm)

  # Balanced accuracy
  balanced_accuracy <- mean(recall)

  # Macro F1
  macro_f1 <- mean(f1)

  class_metrics <- data.frame(
    Class = class_levels,
    Support = as.numeric(rowSums(cm)),
    Precision = as.numeric(precision),
    Recall = as.numeric(recall),
    F1_Score = as.numeric(f1)
  )

  return(list(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    macro_f1 = macro_f1,
    confusion_matrix = cm,
    class_metrics = class_metrics
  ))
}


# ============================================================
# 6. TRAIN AND OPTIMIZE SVM
# ============================================================

run_svm_optimization <- function(task_name, target_col,
                                 train_file, test_file) {

  cat("\n========================================\n")
  cat("SVM OPTIMIZATION:", task_name, "\n")
  cat("========================================\n")

  # Load common train and test datasets
  train_data <- read.csv(
    file.path(SPLIT_DIR, train_file),
    check.names = FALSE
  )

  test_data <- read.csv(
    file.path(SPLIT_DIR, test_file),
    check.names = FALSE
  )

  # Check required columns
  required_cols <- c("File_Name", target_col, feature_cols)

  if (!all(required_cols %in% names(train_data)) ||
      !all(required_cols %in% names(test_data))) {
    stop("Required columns are missing.")
  }

  # Prepare target variables
  y_train <- factor(train_data[[target_col]])

  class_levels <- levels(y_train)

  y_test <- factor(
    test_data[[target_col]],
    levels = class_levels
  )

  if (anyNA(y_train) || anyNA(y_test)) {
    stop("Missing or unseen target classes detected.")
  }

  # Prepare numerical features
  x_train <- train_data[, feature_cols]
  x_test <- test_data[, feature_cols]

  if (any(!is.finite(as.matrix(x_train))) ||
      any(!is.finite(as.matrix(x_test)))) {
    stop("Features contain missing or non-finite values.")
  }

  # Every class needs at least 5 training records
  if (any(table(y_train) < 5)) {
    stop("A class has fewer than 5 training records.")
  }


  # ----------------------------------------------------------
  # 6A. DEFINE HYPERPARAMETER GRID
  # ----------------------------------------------------------

  # Cost controls the penalty for classification errors
  cost_values <- c(0.1, 1, 10, 100, 1000)

  # Gamma controls the influence of individual observations
  gamma_values <- c(0.0001, 0.001, 0.01, 0.1, 1)

  parameter_grid <- expand.grid(
    Cost = cost_values,
    Gamma = gamma_values
  )

  cat("\nTotal combinations:", nrow(parameter_grid), "\n")


  # ----------------------------------------------------------
  # 6B. CREATE COMMON CV FOLDS
  # ----------------------------------------------------------

  set.seed(2026)

  fold_id <- create_folds(y_train, k = 5)

  tuning_results <- data.frame()

  cat("\nStarting 5-fold cross-validation...\n")


  # ----------------------------------------------------------
  # 6C. TEST EACH HYPERPARAMETER COMBINATION
  # ----------------------------------------------------------

  for (i in 1:nrow(parameter_grid)) {

    current_cost <- parameter_grid$Cost[i]
    current_gamma <- parameter_grid$Gamma[i]

    fold_results <- data.frame()

    for (fold in 1:5) {

      # Split training and validation records
      train_indices <- which(fold_id != fold)
      valid_indices <- which(fold_id == fold)

      x_fold_train <- x_train[train_indices, , drop = FALSE]
      x_fold_valid <- x_train[valid_indices, , drop = FALSE]

      y_fold_train <- y_train[train_indices]
      y_fold_valid <- y_train[valid_indices]

      # Calculate weights using only fold training records
      class_weights <- calculate_class_weights(
        y_fold_train,
        class_levels
      )

      # Train SVM
      model <- e1071::svm(
        x = x_fold_train,
        y = y_fold_train,
        type = "C-classification",
        kernel = "radial",
        cost = current_cost,
        gamma = current_gamma,
        class.weights = class_weights,
        scale = TRUE
      )

      # Predict validation records
      predictions <- predict(
        model,
        newdata = x_fold_valid
      )

      # Calculate metrics
      metrics <- calculate_metrics(
        y_fold_valid,
        predictions,
        class_levels
      )

      fold_results <- rbind(
        fold_results,
        data.frame(
          Accuracy = metrics$accuracy,
          Balanced_Accuracy = metrics$balanced_accuracy,
          Macro_F1 = metrics$macro_f1
        )
      )
    }

    # Save average CV performance
    tuning_results <- rbind(
      tuning_results,
      data.frame(
        Cost = current_cost,
        Gamma = current_gamma,
        Mean_Accuracy = mean(fold_results$Accuracy),
        Mean_Balanced_Accuracy =
          mean(fold_results$Balanced_Accuracy),
        Mean_Macro_F1 = mean(fold_results$Macro_F1),
        SD_Macro_F1 = sd(fold_results$Macro_F1)
      )
    )

    cat(
      "Combination:", i,
      "| Cost:", current_cost,
      "| Gamma:", current_gamma,
      "| Accuracy:", round(mean(fold_results$Accuracy), 4),
      "| Macro F1:", round(mean(fold_results$Macro_F1), 4),
      "\n"
    )
  }


  # ----------------------------------------------------------
  # 6D. SELECT BEST HYPERPARAMETERS
  # ----------------------------------------------------------

  # Macro F1 is the primary selection metric.
  # Accuracy and Balanced Accuracy break ties.
  tuning_results <- tuning_results[
    order(
      -tuning_results$Mean_Macro_F1,
      -tuning_results$Mean_Balanced_Accuracy,
      -tuning_results$Mean_Accuracy
    ),
  ]

  best_params <- tuning_results[1, ]

  best_cost <- best_params$Cost
  best_gamma <- best_params$Gamma

  cat("\nBest Cost:", best_cost, "\n")
  cat("Best Gamma:", best_gamma, "\n")
  cat("Best CV Macro F1:", best_params$Mean_Macro_F1, "\n")


  # ----------------------------------------------------------
  # 6E. TRAIN FINAL SVM USING FULL TRAINING DATA
  # ----------------------------------------------------------

  final_weights <- calculate_class_weights(
    y_train,
    class_levels
  )

  final_model <- e1071::svm(
    x = x_train,
    y = y_train,
    type = "C-classification",
    kernel = "radial",
    cost = best_cost,
    gamma = best_gamma,
    class.weights = final_weights,
    scale = TRUE
  )

  # Save optimized model
  saveRDS(
    final_model,
    file.path(
      MODEL_DIR,
      paste0(tolower(task_name), "_optimized_svm.rds")
    )
  )


  # ----------------------------------------------------------
  # 6F. FINAL TEST PREDICTIONS
  # ----------------------------------------------------------

  final_predictions <- predict(
    final_model,
    newdata = x_test
  )

  metrics <- calculate_metrics(
    y_test,
    final_predictions,
    class_levels
  )


  # ----------------------------------------------------------
  # 6G. SAVE PREDICTIONS
  # ----------------------------------------------------------

  prediction_data <- data.frame(
    File_Name = test_data$File_Name,
    Actual = y_test,
    Predicted = final_predictions
  )

  write.csv(
    prediction_data,
    file.path(
      PREDICTION_DIR,
      paste0(tolower(task_name), "_optimized_svm_predictions.csv")
    ),
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # 6H. SAVE CONFUSION MATRIX
  # ----------------------------------------------------------

  write.csv(
    as.data.frame.matrix(metrics$confusion_matrix),
    file.path(
      EVALUATION_DIR,
      paste0(tolower(task_name), "_optimized_svm_confusion.csv")
    )
  )


  # ----------------------------------------------------------
  # 6I. SAVE CLASS-WISE METRICS
  # ----------------------------------------------------------

  write.csv(
    metrics$class_metrics,
    file.path(
      EVALUATION_DIR,
      paste0(tolower(task_name), "_optimized_svm_class_metrics.csv")
    ),
    row.names = FALSE
  )


  # Save tuning results
  write.csv(
    tuning_results,
    file.path(
      EVALUATION_DIR,
      paste0(tolower(task_name), "_optimized_svm_cv_results.csv")
    ),
    row.names = FALSE
  )


  # ----------------------------------------------------------
  # 6J. PRINT FINAL RESULTS
  # ----------------------------------------------------------

  cat("\n========== FINAL TEST RESULTS ==========\n")

  cat("Task:", task_name, "\n")
  cat("Accuracy:", round(metrics$accuracy * 100, 2), "%\n")

  cat(
    "Balanced Accuracy:",
    round(metrics$balanced_accuracy * 100, 2),
    "%\n"
  )

  cat(
    "Macro F1:",
    round(metrics$macro_f1 * 100, 2),
    "%\n"
  )

  cat("\nConfusion Matrix:\n")
  print(metrics$confusion_matrix)

  cat("\nClass-wise Metrics:\n")
  print(metrics$class_metrics)

  # Return results
  return(list(
    tuning = tuning_results,
    summary = data.frame(
      Task = task_name,
      Best_Cost = best_cost,
      Best_Gamma = best_gamma,
      CV_Macro_F1 = best_params$Mean_Macro_F1,
      CV_Accuracy = best_params$Mean_Accuracy,
      Test_Accuracy = metrics$accuracy,
      Test_Balanced_Accuracy = metrics$balanced_accuracy,
      Test_Macro_F1 = metrics$macro_f1
    ),
    class_metrics = metrics$class_metrics
  ))
}


# ============================================================
# 7. OPTIMIZE SPECIES SVM
# ============================================================

species_result <- run_svm_optimization(
  task_name = "Species",
  target_col = "Species",
  train_file = "Species_train.csv",
  test_file = "Species_test.csv"
)


# ============================================================
# 8. OPTIMIZE THREAT SVM
# ============================================================

threat_result <- run_svm_optimization(
  task_name = "Threat",
  target_col = "Threat_Level",
  train_file = "Threat_train.csv",
  test_file = "Threat_test.csv"
)


# ============================================================
# 9. SAVE COMBINED RESULTS
# ============================================================

all_summary <- rbind(
  species_result$summary,
  threat_result$summary
)

all_tuning <- rbind(
  transform(species_result$tuning, Task = "Species"),
  transform(threat_result$tuning, Task = "Threat")
)

all_class_metrics <- rbind(
  transform(species_result$class_metrics, Task = "Species"),
  transform(threat_result$class_metrics, Task = "Threat")
)

write.csv(
  all_summary,
  file.path(EVALUATION_DIR, "optimized_svm_final_summary.csv"),
  row.names = FALSE
)

write.csv(
  all_tuning,
  file.path(EVALUATION_DIR, "optimized_svm_all_cv_results.csv"),
  row.names = FALSE
)

write.csv(
  all_class_metrics,
  file.path(EVALUATION_DIR, "optimized_svm_all_class_metrics.csv"),
  row.names = FALSE
)

cat("\n========================================\n")
cat("SVM OPTIMIZATION COMPLETED!\n")
cat("========================================\n")

print(all_summary)
