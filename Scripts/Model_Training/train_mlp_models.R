
# ============================================================
# MULTILAYER PERCEPTRON (MLP) MODEL
# Species Classification and Threat Level Classification
# ============================================================

# Load required package
if (!requireNamespace("nnet", quietly = TRUE)) {
  install.packages("nnet")
}

# ============================================================
# 1. SET PROJECT PATHS
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
# 2. DEFINE THE 33 AUDIO FEATURES
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
# 3. CREATE STRATIFIED FOLDS
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

  # Check that every training fold contains all classes
  for (fold in 1:k) {

    training_classes <- unique(y[fold_id != fold])

    if (length(training_classes) != length(levels(y))) {
      stop("A training fold is missing one or more classes.")
    }
  }

  return(fold_id)
}



# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

standardize_features <- function(train_x, test_x) {

  # Calculate mean and standard deviation from training data
  mean_values <- colMeans(train_x)
  sd_values <- apply(train_x, 2, sd)

  # Avoid division by zero
  sd_values[!is.finite(sd_values) | sd_values == 0] <- 1

  # Standardize training data
  train_scaled <- scale(
    train_x,
    center = mean_values,
    scale = sd_values
  )

  # Standardize test data using training statistics
  test_scaled <- scale(
    test_x,
    center = mean_values,
    scale = sd_values
  )

  return(list(
    train = train_scaled,
    test = test_scaled,
    mean = mean_values,
    sd = sd_values
  ))
}


# ============================================================
# 5. CALCULATE CLASS WEIGHTS
# ============================================================

calculate_class_weights <- function(y) {

  class_counts <- table(y)

  # Give each class approximately equal total weight
  weights <- length(y) / (
    length(class_counts) *
    as.numeric(class_counts[as.character(y)])
  )

  return(weights)
}


# ============================================================
# 6. CALCULATE EVALUATION METRICS
# ============================================================

calculate_metrics <- function(actual, predicted, class_levels) {

  actual <- factor(actual, levels = class_levels)

  predicted <- factor(predicted, levels = class_levels)

  # Confusion matrix
  cm <- table(
    Actual = actual,
    Predicted = predicted
  )

  # Correct predictions for each class
  correct <- diag(cm)

  # Precision
  precision <- ifelse(
    colSums(cm) == 0,
    0,
    correct / colSums(cm)
  )

  # Recall
  recall <- ifelse(
    rowSums(cm) == 0,
    0,
    correct / rowSums(cm)
  )

  # F1 Score
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
# 7. PREDICT CLASS PROBABILITIES
# ============================================================

get_predictions <- function(model, x, class_levels) {

  # Get probability for every class
  probabilities <- predict(
    model,
    newdata = x,
    type = "raw"
  )

  probabilities <- as.matrix(probabilities)

  # Ensure probability columns match class order
  if (!is.null(colnames(probabilities)) &&
      all(class_levels %in% colnames(probabilities))) {

    probabilities <- probabilities[
      , class_levels, drop = FALSE
    ]

  } else if (ncol(probabilities) == length(class_levels)) {

    colnames(probabilities) <- class_levels

  } else {

    stop("Unexpected number of output probabilities.")
  }

  # Select class with maximum probability
  predicted <- class_levels[
    max.col(probabilities, ties.method = "first")
  ]

  return(list(
    predicted = predicted,
    probabilities = probabilities
  ))
}


# ============================================================
# 8. TRAIN AND EVALUATE ONE TASK
# ============================================================

run_mlp_task <- function(task_name, target_col,
                         train_file, test_file) {

  cat("\n========================================\n")
  cat("MLP:", task_name, "\n")
  cat("========================================\n")

  # Load common train and test splits
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
    stop(paste("Missing required columns for", task_name))
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
  x_train <- as.matrix(train_data[, feature_cols])
  x_test <- as.matrix(test_data[, feature_cols])

  storage.mode(x_train) <- "double"
  storage.mode(x_test) <- "double"

  if (any(!is.finite(x_train)) || any(!is.finite(x_test))) {
    stop("Features contain missing or non-finite values.")
  }

  # Each class must have enough samples for 5-fold CV
  if (any(table(y_train) < 5)) {
    stop("A class has fewer than 5 training records.")
  }

  # ----------------------------------------------------------
  # 8A. HYPERPARAMETER TUNING
  # ----------------------------------------------------------

  hidden_sizes <- c(16, 32, 64)
  decay_values <- c(0.001, 0.01, 0.1)

  k <- 5

  set.seed(2026)

  fold_id <- create_folds(y_train, k)

  tuning_results <- data.frame()

  cat("\nStarting 5-fold cross-validation...\n")

  # Test different hidden layer sizes
  for (hidden_size in hidden_sizes) {

    # Test different weight decay values
    for (decay_value in decay_values) {

      fold_results <- data.frame()

      for (fold in 1:k) {

        # Divide data into fold training and validation sets
        train_indices <- which(fold_id != fold)
        validation_indices <- which(fold_id == fold)

        x_fold_train <- x_train[train_indices, , drop = FALSE]
        x_fold_valid <- x_train[validation_indices, , drop = FALSE]

        y_fold_train <- y_train[train_indices]
        y_fold_valid <- y_train[validation_indices]

        # Standardize using only this fold's training data
        scaled <- standardize_features(
          x_fold_train,
          x_fold_valid
        )

        # Calculate class weights from fold training data
        observation_weights <- calculate_class_weights(
          y_fold_train
        )

        # Train MLP
        
# Convert training labels into one-hot encoding
y_fold_matrix <- model.matrix(~ y_fold_train - 1)

# Set output column names
colnames(y_fold_matrix) <- class_levels

# Train MLP
model <- nnet::nnet(
  x = scaled$train,
  y = y_fold_matrix,
  size = hidden_size,
  decay = decay_value,
  maxit = 500,
  MaxNWts = 20000,
  weights = observation_weights,
  softmax = TRUE,
  trace = FALSE
)


        # Predict validation records
        predictions <- get_predictions(
          model,
          scaled$test,
          class_levels
        )

        # Evaluate validation predictions
        metrics <- calculate_metrics(
          y_fold_valid,
          predictions$predicted,
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

      # Store average cross-validation results
      tuning_results <- rbind(
        tuning_results,
        data.frame(
          Task = task_name,
          Hidden_Units = hidden_size,
          Decay = decay_value,
          Mean_Accuracy = mean(fold_results$Accuracy),
          Mean_Balanced_Accuracy =
            mean(fold_results$Balanced_Accuracy),
          Mean_Macro_F1 = mean(fold_results$Macro_F1),
          SD_Macro_F1 = sd(fold_results$Macro_F1)
        )
      )

      cat(
        "Hidden:", hidden_size,
        "| Decay:", decay_value,
        "| Macro F1:",
        round(mean(fold_results$Macro_F1), 4),
        "\n"
      )
    }
  }

  # ----------------------------------------------------------
  # 8B. SELECT BEST HYPERPARAMETERS
  # ----------------------------------------------------------

  tuning_results <- tuning_results[
    order(
      -tuning_results$Mean_Macro_F1,
      -tuning_results$Mean_Balanced_Accuracy
    ),
  ]

  best_params <- tuning_results[1, ]

  best_hidden <- best_params$Hidden_Units
  best_decay <- best_params$Decay

  cat("\nBest Hidden Units:", best_hidden, "\n")
  cat("Best Decay:", best_decay, "\n")
  cat("Best CV Macro F1:", best_params$Mean_Macro_F1, "\n")

  # ----------------------------------------------------------
  # 8C. TRAIN FINAL MODEL ON FULL TRAINING DATA
  # ----------------------------------------------------------

  # Standardize using the complete training set
  final_scaled <- standardize_features(
    x_train,
    x_test
  )

  # Calculate class weights from full training set
  final_weights <- calculate_class_weights(y_train)

  set.seed(2026)

  
# Convert final training labels into one-hot encoding
y_train_matrix <- model.matrix(~ y_train - 1)

# Set output column names
colnames(y_train_matrix) <- class_levels

# Train final MLP model
final_model <- nnet::nnet(
  x = final_scaled$train,
  y = y_train_matrix,
  size = best_hidden,
  decay = best_decay,
  maxit = 1000,
  MaxNWts = 20000,
  weights = final_weights,
  softmax = TRUE,
  trace = FALSE
)


  # Save model and scaling parameters together
  saveRDS(
    list(
      model = final_model,
      mean = final_scaled$mean,
      sd = final_scaled$sd,
      features = feature_cols,
      classes = class_levels,
      hidden_units = best_hidden,
      decay = best_decay
    ),
    file.path(
      MODEL_DIR,
      paste0(tolower(task_name), "_mlp.rds")
    )
  )

  # ----------------------------------------------------------
  # 8D. FINAL TEST PREDICTIONS
  # ----------------------------------------------------------

  predictions <- get_predictions(
    final_model,
    final_scaled$test,
    class_levels
  )

  predicted_classes <- predictions$predicted

  # Evaluate on untouched common test set
  metrics <- calculate_metrics(
    y_test,
    predicted_classes,
    class_levels
  )

  # ----------------------------------------------------------
  # 8E. SAVE PREDICTIONS
  # ----------------------------------------------------------

  prediction_data <- data.frame(
    File_Name = test_data$File_Name,
    Actual = y_test,
    Predicted = predicted_classes,
    Confidence = apply(
      predictions$probabilities,
      1,
      max
    )
  )

  write.csv(
    prediction_data,
    file.path(
      PREDICTION_DIR,
      paste0(tolower(task_name), "_mlp_predictions.csv")
    ),
    row.names = FALSE
  )

  # ----------------------------------------------------------
  # 8F. SAVE CONFUSION MATRIX
  # ----------------------------------------------------------

  write.csv(
    as.data.frame.matrix(metrics$confusion_matrix),
    file.path(
      EVALUATION_DIR,
      paste0(tolower(task_name), "_mlp_confusion_matrix.csv")
    )
  )

  # ----------------------------------------------------------
  # 8G. SAVE CLASS-WISE METRICS
  # ----------------------------------------------------------

  class_metrics <- metrics$class_metrics

  class_metrics$Task <- task_name

  write.csv(
    class_metrics,
    file.path(
      EVALUATION_DIR,
      paste0(tolower(task_name), "_mlp_class_metrics.csv")
    ),
    row.names = FALSE
  )

  # Save CV tuning results
  write.csv(
    tuning_results,
    file.path(
      EVALUATION_DIR,
      paste0(tolower(task_name), "_mlp_cv_results.csv")
    ),
    row.names = FALSE
  )

  # ----------------------------------------------------------
  # 8H. PRINT FINAL RESULTS
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
  print(class_metrics)

  # Return results for combined summary
  return(list(
    tuning = tuning_results,
    summary = data.frame(
      Task = task_name,
      Best_Hidden_Units = best_hidden,
      Best_Decay = best_decay,
      CV_Macro_F1 = best_params$Mean_Macro_F1,
      Test_Accuracy = metrics$accuracy,
      Test_Balanced_Accuracy = metrics$balanced_accuracy,
      Test_Macro_F1 = metrics$macro_f1
    ),
    class_metrics = class_metrics
  ))
}


# ============================================================
# 9. RUN SPECIES MLP
# ============================================================

species_result <- run_mlp_task(
  task_name = "Species",
  target_col = "Species",
  train_file = "Species_train.csv",
  test_file = "Species_test.csv"
)


# ============================================================
# 10. RUN THREAT MLP
# ============================================================

threat_result <- run_mlp_task(
  task_name = "Threat",
  target_col = "Threat_Level",
  train_file = "Threat_train.csv",
  test_file = "Threat_test.csv"
)


# ============================================================
# 11. SAVE COMBINED RESULTS
# ============================================================

all_tuning_results <- rbind(
  species_result$tuning,
  threat_result$tuning
)

all_summary <- rbind(
  species_result$summary,
  threat_result$summary
)

all_class_metrics <- rbind(
  species_result$class_metrics,
  threat_result$class_metrics
)

write.csv(
  all_tuning_results,
  file.path(EVALUATION_DIR, "mlp_all_cv_results.csv"),
  row.names = FALSE
)

write.csv(
  all_summary,
  file.path(EVALUATION_DIR, "mlp_final_summary.csv"),
  row.names = FALSE
)

write.csv(
  all_class_metrics,
  file.path(EVALUATION_DIR, "mlp_all_class_metrics.csv"),
  row.names = FALSE
)

cat("\n========================================\n")
cat("MLP TRAINING COMPLETED SUCCESSFULLY!\n")
cat("========================================\n")

print(all_summary)
