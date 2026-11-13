
# CatBoost - Threat Detection and Species Classification

library(catboost)

# 1. Define the 33 audio features
features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)

# 2. Function to train and evaluate CatBoost
train_catboost_task <- function(task_name, target_column,
                                train_file, test_file) {

  cat("\n====================================\n")
  cat("TASK:", task_name, "\n")
  cat("====================================\n")

  # Read datasets
  train_data <- read.csv(train_file)
  test_data <- read.csv(test_file)

  # Check required columns
  required <- c(features, target_column)

  if (!all(required %in% names(train_data)) ||
      !all(required %in% names(test_data))) {
    stop("Required columns missing in ", task_name, " dataset.")
  }

  # Remove rows with missing target labels
  train_data <- train_data[
    !is.na(train_data[[target_column]]), ]

  test_data <- test_data[
    !is.na(test_data[[target_column]]), ]

  # Prepare target labels
  train_labels <- as.character(train_data[[target_column]])
  test_labels <- as.character(test_data[[target_column]])

  # Get class names from training data
  class_names <- sort(unique(train_labels))

  # Check for classes not present in training data
  if (any(!test_labels %in% class_names)) {
    stop("Test data contains classes missing from training data.")
  }

  # Convert class labels into numeric values
  y_train <- as.integer(factor(
    train_labels, levels = class_names
  )) - 1

  y_test <- as.integer(factor(
    test_labels, levels = class_names
  )) - 1

  # Prepare numerical features
  X_train <- train_data[, features, drop = FALSE]
  X_test <- test_data[, features, drop = FALSE]

  X_train[] <- lapply(X_train, function(x) {
    suppressWarnings(as.numeric(as.character(x)))
  })

  X_test[] <- lapply(X_test, function(x) {
    suppressWarnings(as.numeric(as.character(x)))
  })

  # Handle missing and infinite values using training medians
  for (col in features) {

    X_train[[col]][!is.finite(X_train[[col]])] <- NA
    X_test[[col]][!is.finite(X_test[[col]])] <- NA

    median_value <- median(X_train[[col]], na.rm = TRUE)

    if (!is.finite(median_value)) {
      stop("Feature contains no valid values: ", col)
    }

    X_train[[col]][is.na(X_train[[col]])] <- median_value
    X_test[[col]][is.na(X_test[[col]])] <- median_value
  }

  # Create CatBoost training and testing pools
  train_pool <- catboost.load_pool(
    data = X_train,
    label = y_train
  )

  test_pool <- catboost.load_pool(
    data = X_test,
    label = y_test
  )

  # CatBoost parameters
  params <- list(
    loss_function = "MultiClass",
    eval_metric = "Accuracy",
    iterations = 600,
    depth = 6,
    learning_rate = 0.05,
    l2_leaf_reg = 5,
    random_seed = 42,
    thread_count = 4,
    verbose = 100,
    allow_writing_files = FALSE
  )

  # Train model
  model <- catboost.train(
    learn_pool = train_pool,
    params = params
  )

  # Predict class probabilities
  probabilities <- catboost.predict(
    model,
    test_pool,
    prediction_type = "Probability"
  )

  # Convert probabilities into predicted classes
  predicted_ids <- max.col(as.matrix(probabilities))
  predicted <- class_names[predicted_ids]

  # Confusion matrix
  cm <- table(
    Actual = factor(test_labels, levels = class_names),
    Predicted = factor(predicted, levels = class_names)
  )

  cat("\nCONFUSION MATRIX:\n")
  print(cm)

  # Calculate metrics
  accuracy <- sum(diag(cm)) / sum(cm)

  precision <- diag(cm) / colSums(cm)
  recall <- diag(cm) / rowSums(cm)

  precision[!is.finite(precision)] <- 0
  recall[!is.finite(recall)] <- 0

  f1 <- 2 * precision * recall / (precision + recall)
  f1[!is.finite(f1)] <- 0

  balanced_accuracy <- mean(recall)
  macro_f1 <- mean(f1)

  cat("\nRESULTS:\n")
  cat("Accuracy:", round(accuracy * 100, 2), "%\n")
  cat("Balanced Accuracy:",
      round(balanced_accuracy * 100, 2), "%\n")
  cat("Macro-F1:", round(macro_f1 * 100, 2), "%\n")

  cat("\nCLASS-WISE METRICS:\n")

  print(data.frame(
    Class = class_names,
    Precision = round(precision, 3),
    Recall = round(recall, 3),
    F1_Score = round(f1, 3)
  ))

  # Save predictions
  write.csv(
    data.frame(
      Actual = test_labels,
      Predicted = predicted
    ),
    paste0("Data_Processed/catboost_",
           tolower(task_name), "_predictions.csv"),
    row.names = FALSE
  )

  # Return metrics
  return(data.frame(
    Task = task_name,
    Accuracy = round(accuracy * 100, 2),
    Balanced_Accuracy = round(balanced_accuracy * 100, 2),
    Macro_F1 = round(macro_f1 * 100, 2)
  ))
}

# 3. Train Threat Classification Model
threat_results <- train_catboost_task(
  task_name = "Threat",
  target_column = "Threat_Level",
  train_file = "Data_Processed/Common_Splits/Threat_train.csv",
  test_file = "Data_Processed/Common_Splits/Threat_test.csv"
)

# 4. Train Species Classification Model
species_results <- train_catboost_task(
  task_name = "Species",
  target_column = "Species",
  train_file = "Data_Processed/Common_Splits/Species_train.csv",
  test_file = "Data_Processed/Common_Splits/Species_test.csv"
)

# 5. Display final comparison
cat("\n========== CATBOOST FINAL RESULTS ==========\n")

final_results <- rbind(threat_results, species_results)

print(final_results)

write.csv(
  final_results,
  "Data_Processed/catboost_final_results.csv",
  row.names = FALSE
)
