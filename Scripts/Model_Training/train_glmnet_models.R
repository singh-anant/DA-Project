# Elastic Net Logistic Regression
# Species and Threat Level Classification

library(glmnet)


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

  cat("\n====================================\n")
  cat(task_name, "- GLMNET RESULTS\n")
  cat("====================================\n")

  cat("Accuracy          :", round(accuracy * 100, 2), "%\n")
  cat("Balanced Accuracy :", round(mean(recall) * 100, 2), "%\n")
  cat("Macro F1 Score    :", round(mean(f1) * 100, 2), "%\n")

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


# Create stratified cross-validation folds
make_folds <- function(y, k = 5) {

  counts <- table(y)

  if (min(counts) < 3) {
    stop(
      "At least 3 training samples per class are required ",
      "for stratified cross-validation."
    )
  }

  k <- min(k, min(counts))
  foldid <- integer(length(y))

  for (cl in levels(y)) {

    index <- which(y == cl)

    foldid[index] <- sample(
      rep(seq_len(k), length.out = length(index))
    )
  }

  return(foldid)
}


# Train and evaluate one classification model
run_glmnet <- function(train_file, test_file, target,
                       task_name, use_class_weights = FALSE) {

  # Load existing datasets
  train <- read.csv(train_file)
  test <- read.csv(test_file)

  # Select 33 audio features
  feature_cols <- grep(
    "^(Pitch_Centroid|Energy|ZCR|Spectral_|MFCC_)",
    names(train),
    value = TRUE
  )

  feature_cols <- intersect(feature_cols, names(test))

  X_train <- as.matrix(train[, feature_cols, drop = FALSE])
  X_test <- as.matrix(test[, feature_cols, drop = FALSE])

  storage.mode(X_train) <- "double"
  storage.mode(X_test) <- "double"

  # Replace missing values using training medians
  medians <- apply(X_train, 2, median, na.rm = TRUE)

  valid <- is.finite(medians)

  X_train <- X_train[, valid, drop = FALSE]
  X_test <- X_test[, valid, drop = FALSE]
  medians <- medians[valid]

  for (j in seq_len(ncol(X_train))) {

    X_train[!is.finite(X_train[, j]), j] <- medians[j]
    X_test[!is.finite(X_test[, j]), j] <- medians[j]
  }

  # Remove constant features
  feature_sd <- apply(X_train, 2, sd)

  valid <- is.finite(feature_sd) & feature_sd > 0

  X_train <- X_train[, valid, drop = FALSE]
  X_test <- X_test[, valid, drop = FALSE]

  # Prepare target labels
  y_train <- factor(train[[target]])

  y_test <- factor(
    test[[target]],
    levels = levels(y_train)
  )

  if (anyNA(y_train) || anyNA(y_test)) {
    stop("Missing or unseen target classes in ", task_name)
  }

  # Generate common folds for all alpha values
  set.seed(42)
  foldid <- make_folds(y_train, k = 5)

  # Apply inverse-frequency weights when required
  sample_weights <- NULL

  if (use_class_weights) {

    class_counts <- table(y_train)

    sample_weights <- 1 / as.numeric(
      class_counts[as.character(y_train)]
    )

    sample_weights <- sample_weights / mean(sample_weights)
  }

  # Alpha values to test
  alpha_values <- c(0, 0.25, 0.5, 0.75, 1)

  cv_models <- list()
  cv_errors <- numeric(length(alpha_values))

  cat("\nTraining", task_name, "...\n")
  cat("Training samples:", nrow(X_train), "\n")
  cat("Testing samples :", nrow(X_test), "\n")
  cat("Audio features  :", ncol(X_train), "\n")

  # Cross-validation for each alpha
  for (i in seq_along(alpha_values)) {

    alpha <- alpha_values[i]

    cat("Testing alpha =", alpha, "\n")

    arguments <- list(
      x = X_train,
      y = y_train,
      family = "multinomial",
      alpha = alpha,
      foldid = foldid,
      type.measure = "class",
      standardize = TRUE,
      maxit = 100000
    )

    if (!is.null(sample_weights)) {
      arguments$weights <- sample_weights
    }

    cv_model <- do.call(cv.glmnet, arguments)

    cv_models[[i]] <- cv_model

    cv_errors[i] <- min(cv_model$cvm)
  }

  # Select alpha with lowest cross-validation error
  best_index <- which.min(cv_errors)

  best_alpha <- alpha_values[best_index]
  best_model <- cv_models[[best_index]]

  cat("\nCross-validation Results:\n")

  print(data.frame(
    Alpha = alpha_values,
    CV_Error = round(cv_errors, 4)
  ))

  cat("\nBest Alpha :", best_alpha, "\n")
  cat("Best Lambda:", best_model$lambda.min, "\n")
  cat("CV Error   :", round(cv_errors[best_index], 4), "\n")

  # Predict using selected model
  predictions <- predict(
    best_model,
    newx = X_test,
    s = "lambda.min",
    type = "class"
  )

  predictions <- factor(
    as.vector(predictions),
    levels = levels(y_train)
  )

  # Final test evaluation
  evaluate_model(y_test, predictions, task_name)
}


# -----------------------------------
# 1. Species Classification
# -----------------------------------

run_glmnet(
  "Data_Processed/Common_Splits/Species_train.csv",
  "Data_Processed/Common_Splits/Species_test.csv",
  "Species",
  "Species Classification",
  use_class_weights = FALSE
)


# -----------------------------------
# 2. Threat Level Classification
# -----------------------------------

run_glmnet(
  "Data_Processed/Common_Splits/Threat_train.csv",
  "Data_Processed/Common_Splits/Threat_test.csv",
  "Threat_Level",
  "Threat Level Classification",
  use_class_weights = TRUE
)