
# ============================================================
# THREAT SVM - CLASS WEIGHT OPTIMIZATION
# Objective: Improve Normal class performance
# ============================================================

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
# 2. LOAD DATA
# ============================================================

train_data <- read.csv(
  file.path(SPLIT_DIR, "Threat_train.csv"),
  check.names = FALSE
)

test_data <- read.csv(
  file.path(SPLIT_DIR, "Threat_test.csv"),
  check.names = FALSE
)

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

# Prepare features
x_train <- train_data[, feature_cols]
x_test <- test_data[, feature_cols]

# Prepare target
y_train <- factor(train_data$Threat_Level)
class_levels <- levels(y_train)

y_test <- factor(
  test_data$Threat_Level,
  levels = class_levels
)

if (anyNA(y_train) || anyNA(y_test)) {
  stop("Missing or unseen threat classes detected.")
}

if (any(!is.finite(as.matrix(x_train))) ||
    any(!is.finite(as.matrix(x_test)))) {
  stop("Features contain missing or non-finite values.")
}

cat("Training records:", nrow(x_train), "\n")
cat("Testing records:", nrow(x_test), "\n")

cat("\nTraining class distribution:\n")
print(table(y_train))


# ============================================================
# 3. CALCULATE BASE CLASS WEIGHTS
# ============================================================

calculate_class_weights <- function(y) {

  class_counts <- table(
    factor(y, levels = class_levels)
  )

  # Inverse-frequency class weights
  weights <- length(y) / (
    length(class_levels) * as.numeric(class_counts)
  )

  names(weights) <- class_levels

  return(weights)
}


# ============================================================
# 4. CREATE STRATIFIED 5-FOLD SPLITS
# ============================================================

create_folds <- function(y, k = 5) {

  fold_id <- integer(length(y))

  for (class_name in levels(y)) {

    # Find records belonging to this class
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
# 5. EVALUATION METRICS
# ============================================================

calculate_metrics <- function(actual, predicted) {

  actual <- factor(actual, levels = class_levels)

  predicted <- factor(predicted, levels = class_levels)

  cm <- table(
    Actual = actual,
    Predicted = predicted
  )

  correct <- diag(cm)

  precision <- ifelse(
    colSums(cm) == 0,
    0,
    correct / colSums(cm)
  )

  recall <- ifelse(
    rowSums(cm) == 0,
    0,
    correct / rowSums(cm)
  )

  f1 <- ifelse(
    precision + recall == 0,
    0,
    2 * precision * recall / (precision + recall)
  )

  return(list(
    accuracy = sum(correct) / sum(cm),
    balanced_accuracy = mean(recall),
    macro_f1 = mean(f1),
    confusion_matrix = cm,
    class_metrics = data.frame(
      Class = class_levels,
      Support = as.numeric(rowSums(cm)),
      Precision = as.numeric(precision),
      Recall = as.numeric(recall),
      F1_Score = as.numeric(f1)
    )
  ))
}


# ============================================================
# 6. FIX SVM HYPERPARAMETERS
# ============================================================

best_cost <- 100
best_gamma <- 0.01

# Test different additional weights for Normal
normal_multipliers <- c(1, 2, 3, 5, 8, 12)

k <- 5

set.seed(2026)

fold_id <- create_folds(y_train, k)

tuning_results <- data.frame()

cat("\nStarting class-weight optimization...\n")


# ============================================================
# 7. TEST DIFFERENT NORMAL CLASS WEIGHTS
# ============================================================

for (multiplier in normal_multipliers) {

  fold_results <- data.frame()

  cat("\nNormal weight multiplier:", multiplier, "\n")

  for (fold in 1:k) {

    # Split training and validation records
    train_indices <- which(fold_id != fold)
    valid_indices <- which(fold_id == fold)

    x_fold_train <- x_train[train_indices, , drop = FALSE]
    x_fold_valid <- x_train[valid_indices, , drop = FALSE]

    y_fold_train <- y_train[train_indices]
    y_fold_valid <- y_train[valid_indices]

    # Calculate weights using fold training records only
    class_weights <- calculate_class_weights(y_fold_train)

    # Increase Normal class weight
    class_weights["Normal"] <-
      class_weights["Normal"] * multiplier

    # Train SVM
    model <- e1071::svm(
      x = x_fold_train,
      y = y_fold_train,
      type = "C-classification",
      kernel = "radial",
      cost = best_cost,
      gamma = best_gamma,
      class.weights = class_weights,
      scale = TRUE
    )

    # Predict validation records
    predictions <- predict(
      model,
      newdata = x_fold_valid
    )

    # Evaluate predictions
    metrics <- calculate_metrics(
      y_fold_valid,
      predictions
    )

    fold_results <- rbind(
      fold_results,
      data.frame(
        Accuracy = metrics$accuracy,
        Balanced_Accuracy = metrics$balanced_accuracy,
        Macro_F1 = metrics$macro_f1,
        Normal_Recall =
          metrics$class_metrics$Recall[
            metrics$class_metrics$Class == "Normal"
          ]
      )
    )
  }

  # Save average CV performance
  tuning_results <- rbind(
    tuning_results,
    data.frame(
      Normal_Multiplier = multiplier,
      Mean_Accuracy = mean(fold_results$Accuracy),
      Mean_Balanced_Accuracy =
        mean(fold_results$Balanced_Accuracy),
      Mean_Macro_F1 = mean(fold_results$Macro_F1),
      Mean_Normal_Recall = mean(fold_results$Normal_Recall),
      SD_Macro_F1 = sd(fold_results$Macro_F1)
    )
  )

  cat(
    "CV Macro F1:",
    round(mean(fold_results$Macro_F1), 4),
    "| Normal Recall:",
    round(mean(fold_results$Normal_Recall), 4),
    "\n"
  )
}


# ============================================================
# 8. SELECT BEST CLASS WEIGHT
# ============================================================

# Primary metric: Macro F1
# Tie-breaker: Balanced Accuracy
# Final tie-breaker: Normal Recall

tuning_results <- tuning_results[
  order(
    -tuning_results$Mean_Macro_F1,
    -tuning_results$Mean_Balanced_Accuracy,
    -tuning_results$Mean_Normal_Recall
  ),
]

best_multiplier <- tuning_results$Normal_Multiplier[1]

cat("\n========================================\n")
cat("BEST NORMAL WEIGHT MULTIPLIER:",
    best_multiplier, "\n")
cat("========================================\n")

print(tuning_results)


# ============================================================
# 9. TRAIN FINAL MODEL
# ============================================================

final_weights <- calculate_class_weights(y_train)

final_weights["Normal"] <-
  final_weights["Normal"] * best_multiplier

cat("\nFinal class weights:\n")
print(final_weights)

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


# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================

saveRDS(
  final_model,
  file.path(
    MODEL_DIR,
    "threat_class_weight_optimized_svm.rds"
  )
)


# ============================================================
# 11. FINAL TEST EVALUATION
# ============================================================

final_predictions <- predict(
  final_model,
  newdata = x_test
)

final_metrics <- calculate_metrics(
  y_test,
  final_predictions
)

cat("\n========== FINAL TEST RESULTS ==========\n")

cat(
  "Accuracy:",
  round(final_metrics$accuracy * 100, 2),
  "%\n"
)

cat(
  "Balanced Accuracy:",
  round(final_metrics$balanced_accuracy * 100, 2),
  "%\n"
)

cat(
  "Macro F1:",
  round(final_metrics$macro_f1 * 100, 2),
  "%\n"
)

cat("\nConfusion Matrix:\n")
print(final_metrics$confusion_matrix)

cat("\nClass-wise Metrics:\n")
print(final_metrics$class_metrics)


# ============================================================
# 12. SAVE OUTPUT FILES
# ============================================================

prediction_data <- data.frame(
  File_Name = test_data$File_Name,
  Actual = y_test,
  Predicted = final_predictions
)

write.csv(
  prediction_data,
  file.path(
    PREDICTION_DIR,
    "threat_class_weight_optimized_predictions.csv"
  ),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(final_metrics$confusion_matrix),
  file.path(
    EVALUATION_DIR,
    "threat_class_weight_optimized_confusion.csv"
  )
)

write.csv(
  final_metrics$class_metrics,
  file.path(
    EVALUATION_DIR,
    "threat_class_weight_optimized_class_metrics.csv"
  ),
  row.names = FALSE
)

write.csv(
  tuning_results,
  file.path(
    EVALUATION_DIR,
    "threat_class_weight_cv_results.csv"
  ),
  row.names = FALSE
)

summary_result <- data.frame(
  Model = "Threat SVM - Class Weight Optimized",
  Cost = best_cost,
  Gamma = best_gamma,
  Normal_Multiplier = best_multiplier,
  Test_Accuracy = final_metrics$accuracy,
  Test_Balanced_Accuracy = final_metrics$balanced_accuracy,
  Test_Macro_F1 = final_metrics$macro_f1
)

write.csv(
  summary_result,
  file.path(
    EVALUATION_DIR,
    "threat_class_weight_optimized_summary.csv"
  ),
  row.names = FALSE
)

cat("\n========================================\n")
cat("CLASS WEIGHT OPTIMIZATION COMPLETED!\n")
cat("========================================\n")

print(summary_result)
