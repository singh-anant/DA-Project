# ============================================================
# STAGE 7: XGBOOST MODEL COMPARISON
#
# Tasks:
# 1. Species Classification
# 2. Threat Level Classification
#
# Dataset: Richer Audio Features
# Features: 33
# Train-Test Split: 80:20
# Cross Validation: 5-Fold
# Evaluation: Accuracy, Balanced Accuracy, Macro-F1
# ============================================================


# ------------------------------------------------------------
# 1. Load Required Libraries
# ------------------------------------------------------------

library(xgboost)
library(caret)
library(dplyr)
library(readr)


# ------------------------------------------------------------
# 2. Define Common Split Folder
# ------------------------------------------------------------

split_folder <- "Data_Processed/Common_Splits"

# ------------------------------------------------------------
# 3. Select Audio Features
# ------------------------------------------------------------

# Select the 7 basic audio features.

basic_features <- c(
  "Pitch_Centroid",
  "Energy",
  "ZCR",
  "Spectral_Bandwidth",
  "Spectral_RollOff85",
  "Spectral_Flatness",
  "Spectral_Entropy"
)

# Select the 13 MFCC mean features.

mfcc_mean_features <- paste0(
  "MFCC_", 1:13, "_Mean"
)

# Select the 13 MFCC standard deviation features.

mfcc_sd_features <- paste0(
  "MFCC_", 1:13, "_SD"
)

# Combine all 33 features.

feature_columns <- c(
  basic_features,
  mfcc_mean_features,
  mfcc_sd_features
)



cat("\nTotal Features Selected:", length(feature_columns), "\n")


# ------------------------------------------------------------
# 4. Function to Calculate Macro-F1
# ------------------------------------------------------------

calculate_macro_f1 <- function(actual, predicted, class_names) {

  # Keep the same class levels for both vectors.

  actual <- factor(actual, levels = class_names)

  predicted <- factor(predicted, levels = class_names)

  f1_scores <- c()

  # Calculate F1-score separately for every class.

  for (class_name in class_names) {

    # True Positives.

    TP <- sum(
      actual == class_name &
        predicted == class_name
    )

    # False Positives.

    FP <- sum(
      actual != class_name &
        predicted == class_name
    )

    # False Negatives.

    FN <- sum(
      actual == class_name &
        predicted != class_name
    )

    # Precision.

    precision <- ifelse(
      TP + FP == 0,
      0,
      TP / (TP + FP)
    )

    # Recall.

    recall <- ifelse(
      TP + FN == 0,
      0,
      TP / (TP + FN)
    )

    # F1-score.

    f1 <- ifelse(
      precision + recall == 0,
      0,
      2 * precision * recall /
        (precision + recall)
    )

    f1_scores <- c(f1_scores, f1)
  }

  # Macro-F1 is the average F1-score of all classes.

  return(mean(f1_scores))
}


# ------------------------------------------------------------
# 5. Function to Prepare Probability Matrix
# ------------------------------------------------------------

get_probability_matrix <- function(predictions, n, k) {

  # Check whether XGBoost already returned a matrix.

  if (!is.null(dim(predictions))) {

    # Preserve the existing matrix structure.

    probability_matrix <- as.matrix(predictions)

    # Handle transposed matrix if returned.

    if (
      nrow(probability_matrix) == k &&
      ncol(probability_matrix) == n
    ) {

      probability_matrix <- t(probability_matrix)
    }

  } else {

    # If predictions are a vector, convert into a matrix.

    if (length(predictions) != n * k) {

      stop("Incorrect number of XGBoost prediction values.")
    }

    probability_matrix <- matrix(
      predictions,
      nrow = n,
      ncol = k,
      byrow = TRUE
    )
  }

  # Verify the final matrix dimensions.

  if (
    nrow(probability_matrix) != n ||
    ncol(probability_matrix) != k
  ) {

    stop(
      paste(
        "Probability matrix dimension mismatch.",
        "Expected:", n, "rows and", k, "columns."
      )
    )
  }

  # Check for invalid prediction values.

  if (any(!is.finite(probability_matrix))) {

    stop("Probability matrix contains invalid values.")
  }

  return(probability_matrix)
}


# ------------------------------------------------------------
# 6. Function to Train and Evaluate XGBoost
# ------------------------------------------------------------

run_xgboost <- function(data, target_column, seed_value) {

  cat("\n\n============================================\n")
  cat("XGBOOST:", target_column, "\n")
  cat("============================================\n")


  # ----------------------------------------------------------
# A. Load Common Train-Test Split
# ----------------------------------------------------------

# The Threat_Level target uses the shorter filename prefix Threat.

split_name <- if (target_column == "Threat_Level") {
  "Threat"
} else {
  target_column
}

train_path <- file.path(
  split_folder,
  paste0(split_name, "_train.csv")
)

test_path <- file.path(
  split_folder,
  paste0(split_name, "_test.csv")
)

train_file_path <- file.path(
  split_folder,
  paste0(split_name, "_train_filenames.csv")
)

test_file_path <- file.path(
  split_folder,
  paste0(split_name, "_test_filenames.csv")
)

train_data <- read_csv(train_path, show_col_types = FALSE)
test_data <- read_csv(test_path, show_col_types = FALSE)

# Check whether all 33 features exist in both datasets.

missing_train <- setdiff(feature_columns, names(train_data))
missing_test <- setdiff(feature_columns, names(test_data))

if (length(missing_train) > 0 || length(missing_test) > 0) {
  stop("Some audio features are missing from common split files.")
}

# Load filenames separately if they are not already present.

if (!("File_Name" %in% names(train_data))) {

  train_files <- read_csv(
    file.path(split_folder,
              paste0(target_column, "_train_filenames.csv")),
    show_col_types = FALSE
  )

  train_data$File_Name <- train_files[[1]]
}

if (!("File_Name" %in% names(test_data))) {

  test_files <- read_csv(
    file.path(split_folder,
              paste0(target_column, "_test_filenames.csv")),
    show_col_types = FALSE
  )

  test_data$File_Name <- test_files[[1]]
}

# Keep only valid records.

train_data <- train_data %>%
  filter(
    !is.na(.data[[target_column]]),
    if_all(all_of(feature_columns), ~ is.finite(.x))
  )

test_data <- test_data %>%
  filter(
    !is.na(.data[[target_column]]),
    if_all(all_of(feature_columns), ~ is.finite(.x))
  )

# Remove unknown labels.

train_data <- train_data %>%
  filter(
    !tolower(trimws(as.character(.data[[target_column]]))) %in%
      c("unknown", "identity unknown")
  )

test_data <- test_data %>%
  filter(
    !tolower(trimws(as.character(.data[[target_column]]))) %in%
      c("unknown", "identity unknown")
  )


# ----------------------------------------------------------
# B. Prepare Class Labels
# ----------------------------------------------------------

class_names <- sort(unique(as.character(train_data[[target_column]])))

number_of_classes <- length(class_names)

# Ensure test data does not contain an unseen class.

if (!all(unique(as.character(test_data[[target_column]])) %in% class_names)) {
  stop("Test data contains a class not present in training data.")
}

y_train <- match(
  as.character(train_data[[target_column]]),
  class_names
) - 1

y_test <- match(
  as.character(test_data[[target_column]]),
  class_names
) - 1

cat("\nTraining Records:", nrow(train_data), "\n")
cat("Testing Records:", nrow(test_data), "\n")
cat("Number of Classes:", number_of_classes, "\n")

cat("\nTraining Class Distribution:\n")
print(table(train_data[[target_column]]))

cat("\nTesting Class Distribution:\n")
print(table(test_data[[target_column]]))


# ----------------------------------------------------------
# C. Verify Common Split
# ----------------------------------------------------------

common_filenames <- intersect(
  train_data$File_Name,
  test_data$File_Name
)

cat("\nFilename Overlap:", length(common_filenames), "\n")

if (length(common_filenames) > 0) {
  stop("Data leakage detected!")
}
  # ----------------------------------------------------------
  # D. Convert Class Labels into Numeric Values
  # ----------------------------------------------------------

  # XGBoost requires class labels as integers:
  # 0, 1, 2, 3, ...

  y_train <- match(
    as.character(train_data[[target_column]]),
    class_names
  ) - 1

  y_test <- match(
    as.character(test_data[[target_column]]),
    class_names
  ) - 1


  # ----------------------------------------------------------
  # E. Prepare Feature Matrices
  # ----------------------------------------------------------

  x_train <- as.matrix(
    train_data[, feature_columns, drop = FALSE]
  )

  x_test <- as.matrix(
    test_data[, feature_columns, drop = FALSE]
  )

  # Ensure feature values are numeric.

  storage.mode(x_train) <- "double"

  storage.mode(x_test) <- "double"


  # ----------------------------------------------------------
  # F. Create XGBoost DMatrix
  # ----------------------------------------------------------

  dtrain <- xgb.DMatrix(
    data = x_train,
    label = y_train
  )

  dtest <- xgb.DMatrix(
    data = x_test,
    label = y_test
  )


  # ----------------------------------------------------------
  # G. Define Hyperparameter Combinations
  # ----------------------------------------------------------

  # Test different combinations of:
  # nrounds, max_depth and eta.

  xgb_grid <- expand.grid(

    nrounds = c(100, 200),

    max_depth = c(3, 6),

    eta = c(0.05, 0.10),

    KEEP.OUT.ATTRS = FALSE
  )

  cat(
    "\nTotal Parameter Combinations:",
    nrow(xgb_grid),
    "\n"
  )


  # ----------------------------------------------------------
  # H. Perform 5-Fold Cross Validation
  # ----------------------------------------------------------

  set.seed(seed_value)

  folds <- createFolds(
    factor(y_train),
    k = 5,
    list = TRUE,
    returnTrain = FALSE
  )

  # Store cross-validation results.

  cv_results <- data.frame()

  best_macro_f1 <- -1

  best_parameters <- NULL


  # Test each hyperparameter combination.

  for (grid_index in 1:nrow(xgb_grid)) {

    current_parameters <- xgb_grid[grid_index, ]

    fold_scores <- c()

    cat(
      "\nTesting Combination:",
      grid_index,
      "of",
      nrow(xgb_grid),
      "\n"
    )


    # Perform training on each fold.

    for (fold_index in seq_along(folds)) {

      validation_indices <- folds[[fold_index]]

      training_indices <- setdiff(
        seq_along(y_train),
        validation_indices
      )


      # Create training data for the current fold.

      fold_train <- xgb.DMatrix(

        data = x_train[
          training_indices,
          ,
          drop = FALSE
        ],

        label = y_train[training_indices]
      )


      # Create validation data for the current fold.

      fold_validation <- xgb.DMatrix(

        data = x_train[
          validation_indices,
          ,
          drop = FALSE
        ]
      )


      # Train the XGBoost model.

      fold_model <- xgb.train(

        params = list(

          objective = "multi:softprob",

          num_class = number_of_classes,

          eval_metric = "mlogloss",

          max_depth = current_parameters$max_depth,

          eta = current_parameters$eta,

          gamma = 0,

          colsample_bytree = 0.8,

          min_child_weight = 1,

          subsample = 0.8,

          nthread = 1,

          tree_method = "hist",

          seed = seed_value + fold_index
        ),

        data = fold_train,

        nrounds = current_parameters$nrounds,

        verbose = 0
      )


      # ------------------------------------------------------
      # Predict Validation Data
      # ------------------------------------------------------

      probabilities <- predict(
        fold_model,
        fold_validation
      )


      # Convert predictions into the correct matrix format.

      probability_matrix <- get_probability_matrix(

        predictions = probabilities,

        n = length(validation_indices),

        k = number_of_classes
      )


      # Select the class with the highest probability.

      predicted_labels <- max.col(
        probability_matrix,
        ties.method = "first"
      ) - 1


      # Convert numeric predictions into class names.

      actual_classes <- class_names[
        y_train[validation_indices] + 1
      ]

      predicted_classes <- class_names[
        predicted_labels + 1
      ]


      # Calculate Macro-F1.

      fold_f1 <- calculate_macro_f1(

        actual = actual_classes,

        predicted = predicted_classes,

        class_names = class_names
      )

      fold_scores <- c(fold_scores, fold_f1)
    }


    # Calculate average Macro-F1 across five folds.

    mean_f1 <- mean(fold_scores, na.rm = TRUE)

    cat(
      "Mean CV Macro-F1:",
      round(mean_f1, 4),
      "\n"
    )


    # Save the current parameter results.

    cv_results <- rbind(

      cv_results,

      data.frame(

        nrounds = current_parameters$nrounds,

        max_depth = current_parameters$max_depth,

        eta = current_parameters$eta,

        Macro_F1 = mean_f1
      )
    )


    # Select the best parameter combination.

    if (!is.na(mean_f1) && mean_f1 > best_macro_f1) {

      best_macro_f1 <- mean_f1

      best_parameters <- current_parameters
    }
  }


  # Stop if no valid parameters were found.

  if (is.null(best_parameters)) {

    stop("Unable to find valid XGBoost hyperparameters.")
  }


  # ----------------------------------------------------------
  # I. Display Best Hyperparameters
  # ----------------------------------------------------------

  cat("\nBest Hyperparameters:\n")

  print(best_parameters)

  cat(
    "\nBest Cross-Validation Macro-F1:",
    round(best_macro_f1, 4),
    "\n"
  )


  # ----------------------------------------------------------
  # J. Train Final XGBoost Model
  # ----------------------------------------------------------

  final_model <- xgb.train(

    params = list(

      objective = "multi:softprob",

      num_class = number_of_classes,

      eval_metric = "mlogloss",

      max_depth = best_parameters$max_depth,

      eta = best_parameters$eta,

      gamma = 0,

      colsample_bytree = 0.8,

      min_child_weight = 1,

      subsample = 0.8,

      nthread = 1,

      tree_method = "hist",

      seed = seed_value
    ),

    data = dtrain,

    nrounds = best_parameters$nrounds,

    verbose = 0
  )


  # ----------------------------------------------------------
  # K. Predict Test Data
  # ----------------------------------------------------------

  probabilities <- predict(
    final_model,
    dtest
  )


  # Convert predictions into the correct matrix format.

  probability_matrix <- get_probability_matrix(

    predictions = probabilities,

    n = length(y_test),

    k = number_of_classes
  )


  # Select the class having the highest probability.

  predicted_labels <- max.col(
    probability_matrix,
    ties.method = "first"
  ) - 1


  # Convert numeric predictions into class names.

  predicted_classes <- factor(

    class_names[predicted_labels + 1],

    levels = class_names
  )

  actual_classes <- factor(

    class_names[y_test + 1],

    levels = class_names
  )


  # ----------------------------------------------------------
  # L. Calculate Evaluation Metrics
  # ----------------------------------------------------------

  confusion <- confusionMatrix(

    data = predicted_classes,

    reference = actual_classes
  )


  # Accuracy.

  accuracy <- as.numeric(
    confusion$overall["Accuracy"]
  )


  # Balanced Accuracy.

  balanced_accuracy <- mean(
    confusion$byClass[, "Balanced Accuracy"],
    na.rm = TRUE
  )


  # Macro-F1.

  macro_f1 <- calculate_macro_f1(

    actual = actual_classes,

    predicted = predicted_classes,

    class_names = class_names
  )


  # ----------------------------------------------------------
  # M. Display Final Results
  # ----------------------------------------------------------

  cat("\n\n========== FINAL TEST RESULTS ==========\n")

  cat("Target:", target_column, "\n")

  cat("Accuracy:",
      round(accuracy * 100, 2), "%\n")

  cat("Balanced Accuracy:",
      round(balanced_accuracy * 100, 2), "%\n")

  cat("Macro-F1:",
      round(macro_f1 * 100, 2), "%\n")


  # ----------------------------------------------------------
  # N. Display and Save Confusion Matrix
  # ----------------------------------------------------------

  cat("\nConfusion Matrix:\n")

  print(confusion)


  write.csv(

    as.matrix(confusion$table),

    paste0(
      "Data_Processed/xgboost_",
      target_column,
      "_confusion_matrix.csv"
    ),

    row.names = TRUE
  )


  # ----------------------------------------------------------
  # O. Save Class-wise Metrics
  # ----------------------------------------------------------

  class_metrics <- as.data.frame(
    confusion$byClass
  )

  class_metrics$Class <- rownames(class_metrics)

  write.csv(

    class_metrics,

    paste0(
      "Data_Processed/xgboost_",
      target_column,
      "_class_metrics.csv"
    ),

    row.names = FALSE
  )


  # ----------------------------------------------------------
  # P. Save Cross-Validation Results
  # ----------------------------------------------------------

  write.csv(

    cv_results,

    paste0(
      "Data_Processed/xgboost_",
      target_column,
      "_cv_results.csv"
    ),

    row.names = FALSE
  )


  # ----------------------------------------------------------
  # Q. Save Test Predictions
  # ----------------------------------------------------------

  prediction_results <- data.frame(

    Actual = actual_classes,

    Predicted = predicted_classes
  )

  # Include filenames to verify test-set consistency.

  if ("File_Name" %in% names(test_data)) {

    prediction_results <- data.frame(

      File_Name = test_data$File_Name,

      prediction_results
    )
  }

  write.csv(

    prediction_results,

    paste0(
      "Data_Processed/xgboost_",
      target_column,
      "_predictions.csv"
    ),

    row.names = FALSE
  )


  # ----------------------------------------------------------
  # R. Save Trained Model
  # ----------------------------------------------------------

  # Save model and supporting information as an RDS file.

  saveRDS(

    list(

      model = final_model,

      features = feature_columns,

      class_names = class_names,

      best_parameters = best_parameters,

      target_column = target_column
    ),

    paste0(
      "Data_Processed/xgboost_",
      target_column,
      "_model.rds"
    )
  )


  # Save native XGBoost model in UBJSON format.

  xgb.save(

    final_model,

    paste0(
      "Data_Processed/xgboost_",
      target_column,
      "_model.ubj"
    )
  )


  # ----------------------------------------------------------
  # S. Return Final Summary
  # ----------------------------------------------------------

  return(

    data.frame(

      Algorithm = "XGBoost",

      Target = target_column,

      Accuracy = round(accuracy * 100, 2),

      Balanced_Accuracy = round(
        balanced_accuracy * 100, 2
      ),

      Macro_F1 = round(macro_f1 * 100, 2)
    )
  )
}


# ------------------------------------------------------------
# 7. Run Species Classification
# ------------------------------------------------------------

species_results <- run_xgboost(

  data = data,

  target_column = "Species",

  seed_value = 42
)


# ------------------------------------------------------------
# 8. Run Threat Level Classification
# ------------------------------------------------------------

threat_results <- run_xgboost(

  data = data,

  target_column = "Threat_Level",

  seed_value = 43
)


# ------------------------------------------------------------
# 9. Display Final XGBoost Results
# ------------------------------------------------------------

final_results <- rbind(

  species_results,

  threat_results
)

cat("\n\n============================================\n")

cat("FINAL XGBOOST RESULTS\n")

cat("============================================\n")

print(final_results)


# ------------------------------------------------------------
# 10. Save Final Summary
# ------------------------------------------------------------

write.csv(

  final_results,

  "Data_Processed/xgboost_final_summary.csv",

  row.names = FALSE
)

cat("\nAll XGBoost results saved successfully!\n")