# Shrinkage Discriminant Analysis (SDA)
# Train Species and Threat Level Classification models

library(sda)

# Calculate and print classification metrics
evaluate_model <- function(actual, predicted, task_name) {

  classes <- levels(actual)

  cm <- table(
    Actual = factor(actual, levels = classes),
    Predicted = factor(predicted, levels = classes)
  )

  accuracy <- sum(diag(cm)) / sum(cm)

  precision <- diag(cm) / pmax(colSums(cm), 1)
  recall <- diag(cm) / pmax(rowSums(cm), 1)

  f1 <- ifelse(
    precision + recall > 0,
    2 * precision * recall / (precision + recall),
    0
  )

  balanced_accuracy <- mean(recall)
  macro_f1 <- mean(f1)

  cat("\n====================================\n")
  cat(task_name, " - SDA RESULTS\n")
  cat("====================================\n")

  cat("Accuracy          :", round(accuracy * 100, 2), "%\n")
  cat("Balanced Accuracy :", round(balanced_accuracy * 100, 2), "%\n")
  cat("Macro F1 Score    :", round(macro_f1 * 100, 2), "%\n")

  cat("\nConfusion Matrix:\n")
  print(cm)

  cat("\nClass-wise Metrics:\n")

  metrics <- data.frame(
    Class = classes,
    Precision = round(precision, 3),
    Recall = round(recall, 3),
    F1_Score = round(f1, 3)
  )

  print(metrics, row.names = FALSE)
}


# Train and evaluate one SDA model
run_sda <- function(train_file, test_file, target, task_name) {

  # Load existing train and test datasets
  train <- read.csv(train_file)
  test <- read.csv(test_file)

  # Select the 33 audio features
  feature_cols <- grep(
    "^(Pitch_Centroid|Energy|ZCR|Spectral_|MFCC_)",
    names(train),
    value = TRUE
  )

  feature_cols <- intersect(feature_cols, names(test))

  # Extract feature matrices
  X_train <- as.matrix(train[, feature_cols, drop = FALSE])
  X_test <- as.matrix(test[, feature_cols, drop = FALSE])

  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"

  # Calculate missing-value replacements using training data only
  medians <- apply(X_train, 2, median, na.rm = TRUE)

  # Remove features that contain no valid training values
  valid <- is.finite(medians)

  X_train <- X_train[, valid, drop = FALSE]
  X_test <- X_test[, valid, drop = FALSE]
  medians <- medians[valid]

  # Replace missing or infinite values
  for (j in seq_len(ncol(X_train))) {

    X_train[!is.finite(X_train[, j]), j] <- medians[j]
    X_test[!is.finite(X_test[, j]), j] <- medians[j]
  }

  # Remove constant features
  feature_sd <- apply(X_train, 2, sd)

  valid <- is.finite(feature_sd) & feature_sd > 0

  X_train <- X_train[, valid, drop = FALSE]
  X_test <- X_test[, valid, drop = FALSE]

  # Standardize using training data statistics
  train_mean <- colMeans(X_train)
  train_sd <- apply(X_train, 2, sd)

  X_train <- scale(X_train, center = train_mean, scale = train_sd)

  X_test <- sweep(X_test, 2, train_mean, "-")
  X_test <- sweep(X_test, 2, train_sd, "/")

  # Prepare target labels
  y_train <- factor(train[[target]])

  y_test <- factor(
    test[[target]],
    levels = levels(y_train)
  )

  if (anyNA(y_train) || anyNA(y_test)) {
    stop("Missing or unseen target classes found in ", task_name)
  }

  cat("\nTraining", task_name, "SDA Model...\n")
  cat("Training samples:", nrow(X_train), "\n")
  cat("Testing samples :", nrow(X_test), "\n")
  cat("Audio features   :", ncol(X_train), "\n")

  # Train SDA with automatically estimated shrinkage
  sda_model <- sda(
    Xtrain = X_train,
    L = y_train,
    verbose = FALSE
  )

  # Predict test samples
  predictions <- predict(
    sda_model,
    X_test,
    verbose = FALSE
  )$class

  # Evaluate model
  evaluate_model(y_test, predictions, task_name)
}


# -----------------------------------
# 1. Species Classification
# -----------------------------------

run_sda(
  "Data_Processed/Common_Splits/Species_train.csv",
  "Data_Processed/Common_Splits/Species_test.csv",
  "Species",
  "Species Classification"
)


# -----------------------------------
# 2. Threat Level Classification
# -----------------------------------

run_sda(
  "Data_Processed/Common_Splits/Threat_train.csv",
  "Data_Processed/Common_Splits/Threat_test.csv",
  "Threat_Level",
  "Threat Level Classification"
)