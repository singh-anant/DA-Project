# XGBoost Optimization - Species and Threat Classification
# Uses the existing common train/test splits and all 33 audio features.
# Hyperparameters are selected by 5-fold cross-validation Accuracy.
# The held-out test sets are used only for the final evaluation.

library(xgboost)
library(caret)

# 1. Common split folder
split_folder <- "Data_Processed/Common_Splits"

# 2. Select the same 33 audio features
basic_features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy"
)

mfcc_mean_features <- paste0("MFCC_", 1:13, "_Mean")
mfcc_sd_features <- paste0("MFCC_", 1:13, "_SD")
feature_columns <- c(basic_features, mfcc_mean_features, mfcc_sd_features)

if (length(feature_columns) != 33) {
  stop("Expected exactly 33 audio features.")
}

# 3. Convert probability output into an n x k matrix
get_probability_matrix <- function(predictions, n, k) {
  if (!is.null(dim(predictions))) {
    probability_matrix <- as.matrix(predictions)
    if (nrow(probability_matrix) == k && ncol(probability_matrix) == n) {
      probability_matrix <- t(probability_matrix)
    }
  } else {
    if (length(predictions) != n * k) {
      stop("Incorrect number of XGBoost prediction values.")
    }
    probability_matrix <- matrix(predictions, nrow = n, ncol = k, byrow = TRUE)
  }

  if (nrow(probability_matrix) != n || ncol(probability_matrix) != k) {
    stop("Probability matrix dimensions are incorrect.")
  }
  if (any(!is.finite(probability_matrix))) {
    stop("Probability matrix contains non-finite values.")
  }
  probability_matrix
}

# 4. Calculate metrics consistently across all model families
calculate_metrics <- function(actual_ids, predicted_ids, class_names) {
  actual <- factor(class_names[actual_ids + 1], levels = class_names)
  predicted <- factor(class_names[predicted_ids + 1], levels = class_names)
  cm <- table(Actual = actual, Predicted = predicted)

  tp <- diag(cm)
  predicted_count <- colSums(cm)
  actual_count <- rowSums(cm)

  precision <- ifelse(predicted_count > 0, tp / predicted_count, 0)
  recall <- ifelse(actual_count > 0, tp / actual_count, 0)
  f1 <- ifelse(precision + recall > 0,
               2 * precision * recall / (precision + recall), 0)

  metrics <- data.frame(
    Accuracy = sum(tp) / sum(cm),
    Balanced_Accuracy = mean(recall),  # Macro-average recall
    Macro_F1 = mean(f1)
  )

  class_metrics <- data.frame(
    Class = class_names,
    Precision = precision,
    Recall = recall,
    F1_Score = f1,
    Support = as.numeric(actual_count)
  )

  list(metrics = metrics, confusion = cm, class_metrics = class_metrics)
}

# 5. Train and tune one classification task
run_xgboost <- function(target_column, seed_value = 42) {
  cat("\n==================================================\n")
  cat("XGBOOST ACCURACY OPTIMIZATION:", target_column, "\n")
  cat("==================================================\n")

  split_name <- if (target_column == "Threat_Level") "Threat" else target_column
  train_path <- file.path(split_folder, paste0(split_name, "_train.csv"))
  test_path <- file.path(split_folder, paste0(split_name, "_test.csv"))

  if (!file.exists(train_path) || !file.exists(test_path)) {
    stop("Could not find common split files for: ", target_column)
  }

  train_data <- read.csv(train_path, check.names = FALSE)
  test_data <- read.csv(test_path, check.names = FALSE)

  required_columns <- c(target_column, feature_columns)
  missing_train <- setdiff(required_columns, names(train_data))
  missing_test <- setdiff(required_columns, names(test_data))
  if (length(missing_train) > 0 || length(missing_test) > 0) {
    stop("Required target/features are missing from the split files.")
  }

  # Ensure features are numeric
  for (feature in feature_columns) {
    train_data[[feature]] <- suppressWarnings(as.numeric(as.character(train_data[[feature]])))
    test_data[[feature]] <- suppressWarnings(as.numeric(as.character(test_data[[feature]])))
  }

  # Remove missing labels, unknown labels, and rows with invalid audio features
  train_labels <- trimws(as.character(train_data[[target_column]]))
  test_labels <- trimws(as.character(test_data[[target_column]]))

  train_keep <- !is.na(train_labels) &
    !tolower(train_labels) %in% c("unknown", "identity unknown") &
    apply(train_data[, feature_columns, drop = FALSE], 1, function(x) all(is.finite(x)))

  test_keep <- !is.na(test_labels) &
    !tolower(test_labels) %in% c("unknown", "identity unknown") &
    apply(test_data[, feature_columns, drop = FALSE], 1, function(x) all(is.finite(x)))

  train_data <- train_data[train_keep, , drop = FALSE]
  test_data <- test_data[test_keep, , drop = FALSE]
  train_labels <- train_labels[train_keep]
  test_labels <- test_labels[test_keep]

  class_names <- sort(unique(train_labels))
  number_of_classes <- length(class_names)

  if (number_of_classes < 2) stop("Fewer than two training classes found.")
  if (any(!test_labels %in% class_names)) {
    stop("Test data contains a class not present in training data.")
  }

  y_train <- match(train_labels, class_names) - 1L
  y_test <- match(test_labels, class_names) - 1L

  x_train <- as.matrix(train_data[, feature_columns, drop = FALSE])
  x_test <- as.matrix(test_data[, feature_columns, drop = FALSE])
  storage.mode(x_train) <- "double"
  storage.mode(x_test) <- "double"

  cat("Training records:", nrow(x_train), "\n")
  cat("Testing records:", nrow(x_test), "\n")
  cat("Classes:", number_of_classes, "\n")
  cat("Training class distribution:\n")
  print(table(train_labels))

  # Check that no filename is shared between train and test
  if ("File_Name" %in% names(train_data) && "File_Name" %in% names(test_data)) {
    overlap <- intersect(train_data$File_Name, test_data$File_Name)
    cat("Filename overlap:", length(overlap), "\n")
    if (length(overlap) > 0) stop("Data leakage: filenames overlap across splits.")
  }

  dtrain_full <- xgb.DMatrix(data = x_train, label = y_train)
  dtest <- xgb.DMatrix(data = x_test, label = y_test)

  # Stratified 5-fold cross-validation
  set.seed(seed_value)
  folds <- createFolds(factor(y_train), k = 5, list = TRUE, returnTrain = FALSE)

  # Compact but broader grid than the original 8 configurations
  parameter_grid <- expand.grid(
    nrounds = c(100, 200),
    max_depth = c(3, 5),
    eta = c(0.03, 0.10),
    subsample = c(0.8, 1.0),
    KEEP.OUT.ATTRS = FALSE
  )

  cv_results <- vector("list", nrow(parameter_grid))

  for (grid_index in seq_len(nrow(parameter_grid))) {
    config <- parameter_grid[grid_index, ]
    oof_predictions <- rep(NA_integer_, length(y_train))

    cat("\nConfiguration", grid_index, "of", nrow(parameter_grid),
        "| rounds:", config$nrounds,
        "depth:", config$max_depth,
        "eta:", config$eta,
        "subsample:", config$subsample, "\n")

    for (fold_index in seq_along(folds)) {
      validation_indices <- folds[[fold_index]]
      training_indices <- setdiff(seq_along(y_train), validation_indices)

      fold_train <- xgb.DMatrix(
        data = x_train[training_indices, , drop = FALSE],
        label = y_train[training_indices]
      )
      fold_valid <- xgb.DMatrix(
        data = x_train[validation_indices, , drop = FALSE]
      )

      fold_model <- xgb.train(
        params = list(
          objective = "multi:softprob",
          num_class = number_of_classes,
          eval_metric = "mlogloss",
          max_depth = config$max_depth,
          eta = config$eta,
          gamma = 0,
          min_child_weight = 1,
          subsample = config$subsample,
          colsample_bytree = 0.8,
          lambda = 1,
          nthread = 4,
          tree_method = "hist",
          seed = seed_value + fold_index
        ),
        data = fold_train,
        nrounds = config$nrounds,
        verbose = 0
      )

      probabilities <- predict(fold_model, fold_valid)
      probability_matrix <- get_probability_matrix(
        probabilities, length(validation_indices), number_of_classes
      )
      oof_predictions[validation_indices] <- max.col(
        probability_matrix, ties.method = "first"
      ) - 1L
    }

    cv_eval <- calculate_metrics(y_train, oof_predictions, class_names)
    cv_results[[grid_index]] <- data.frame(
      nrounds = config$nrounds,
      max_depth = config$max_depth,
      eta = config$eta,
      subsample = config$subsample,
      Accuracy = cv_eval$metrics$Accuracy,
      Balanced_Accuracy = cv_eval$metrics$Balanced_Accuracy,
      Macro_F1 = cv_eval$metrics$Macro_F1
    )

    cat("CV Accuracy:", round(cv_eval$metrics$Accuracy * 100, 2), "% |",
        "Macro-F1:", round(cv_eval$metrics$Macro_F1 * 100, 2), "%\n")
  }

  cv_results <- do.call(rbind, cv_results)
  cv_results <- cv_results[order(
    -cv_results$Accuracy,
    -cv_results$Macro_F1,
    -cv_results$Balanced_Accuracy
  ), ]

  best_parameters <- cv_results[1, ]

  cat("\nTop CV configurations:\n")
  print(head(cv_results, 5))
  cat("\nSelected parameters (highest CV Accuracy):\n")
  print(best_parameters)

  # Train the selected model on the complete training split
  final_model <- xgb.train(
    params = list(
      objective = "multi:softprob",
      num_class = number_of_classes,
      eval_metric = "mlogloss",
      max_depth = best_parameters$max_depth,
      eta = best_parameters$eta,
      gamma = 0,
      min_child_weight = 1,
      subsample = best_parameters$subsample,
      colsample_bytree = 0.8,
      lambda = 1,
      nthread = 4,
      tree_method = "hist",
      seed = seed_value
    ),
    data = dtrain_full,
    nrounds = best_parameters$nrounds,
    verbose = 0
  )

  test_probabilities <- predict(final_model, dtest)
  test_probability_matrix <- get_probability_matrix(
    test_probabilities, length(y_test), number_of_classes
  )
  test_predictions <- max.col(test_probability_matrix, ties.method = "first") - 1L

  test_eval <- calculate_metrics(y_test, test_predictions, class_names)

  cat("\n========== FINAL TEST RESULTS ==========\n")
  cat("Task:", target_column, "\n")
  cat("Accuracy:", round(test_eval$metrics$Accuracy * 100, 2), "%\n")
  cat("Balanced Accuracy:",
      round(test_eval$metrics$Balanced_Accuracy * 100, 2), "%\n")
  cat("Macro-F1:", round(test_eval$metrics$Macro_F1 * 100, 2), "%\n")
  cat("\nConfusion Matrix (rows=actual, columns=predicted):\n")
  print(test_eval$confusion)
  cat("\nClass-wise metrics:\n")
  print(test_eval$class_metrics)

  # Save all outputs
  output_prefix <- file.path("Data_Processed", paste0("xgboost_optimized_", split_name))

  write.csv(cv_results, paste0(output_prefix, "_cv_results.csv"), row.names = FALSE)
  write.csv(as.data.frame.matrix(test_eval$confusion),
            paste0(output_prefix, "_confusion_matrix.csv"))
  write.csv(test_eval$class_metrics,
            paste0(output_prefix, "_class_metrics.csv"), row.names = FALSE)
  write.csv(data.frame(
    Actual = class_names[y_test + 1],
    Predicted = class_names[test_predictions + 1]
  ), paste0(output_prefix, "_predictions.csv"), row.names = FALSE)
  saveRDS(list(
    model = final_model,
    features = feature_columns,
    class_names = class_names,
    best_parameters = best_parameters,
    target_column = target_column
  ), paste0(output_prefix, "_model.rds"))

  summary <- data.frame(
    Task = target_column,
    Accuracy = round(test_eval$metrics$Accuracy * 100, 2),
    Balanced_Accuracy = round(test_eval$metrics$Balanced_Accuracy * 100, 2),
    Macro_F1 = round(test_eval$metrics$Macro_F1 * 100, 2),
    CV_Accuracy = round(best_parameters$Accuracy * 100, 2)
  )

  summary
}

# 6. Run both classification tasks
species_results <- run_xgboost("Species", seed_value = 42)
threat_results <- run_xgboost("Threat_Level", seed_value = 43)

final_results <- rbind(species_results, threat_results)

cat("\n========== OPTIMIZED XGBOOST SUMMARY ==========\n")
print(final_results)

write.csv(final_results,
          "Data_Processed/xgboost_accuracy_optimized_summary.csv",
          row.names = FALSE)
cat("\nResults saved in Data_Processed/.\n")
