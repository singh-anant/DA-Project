
# Extra Trees - Threat and Species Classification

library(ranger)

# 1. Define 33 audio features
features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)

# 2. Function to train and evaluate Extra Trees
train_extra_trees <- function(task_name, target,
                              train_file, test_file) {

  cat("\n====================================\n")
  cat("TASK:", task_name, "\n")
  cat("====================================\n")

  # Read datasets
  train_data <- read.csv(train_file)
  test_data <- read.csv(test_file)

  # Check columns
  if (!all(c(features, target) %in% names(train_data)) ||
      !all(c(features, target) %in% names(test_data))) {
    stop("Required columns are missing.")
  }

  # Remove missing target labels
  train_data <- train_data[!is.na(train_data[[target]]), ]
  test_data <- test_data[!is.na(test_data[[target]]), ]

  # Get class names
  class_names <- sort(unique(as.character(train_data[[target]])))

  y_train <- factor(
    as.character(train_data[[target]]),
    levels = class_names
  )

  y_test <- as.character(test_data[[target]])

  # Check unseen classes
  if (any(!y_test %in% class_names)) {
    stop("Test contains classes missing from training data.")
  }

  # Prepare features
  X_train <- train_data[, features, drop = FALSE]
  X_test <- test_data[, features, drop = FALSE]

  X_train[] <- lapply(X_train, as.numeric)
  X_test[] <- lapply(X_test, as.numeric)

  # Handle missing and infinite values
  for (col in features) {

    X_train[[col]][!is.finite(X_train[[col]])] <- NA
    X_test[[col]][!is.finite(X_test[[col]])] <- NA

    median_value <- median(X_train[[col]], na.rm = TRUE)

    if (!is.finite(median_value)) {
      stop("Invalid feature: ", col)
    }

    X_train[[col]][is.na(X_train[[col]])] <- median_value
    X_test[[col]][is.na(X_test[[col]])] <- median_value
  }

  # Combine features and target
  model_data <- X_train
  model_data$Target <- y_train

  # Train Extra Trees
  model <- ranger(
    Target ~ .,
    data = model_data,
    num.trees = 800,
    mtry = 6,
    min.node.size = 2,
    splitrule = "extratrees",
    num.random.splits = 5,
    replace = FALSE,
    sample.fraction = 1,
    probability = TRUE,
    num.threads = 4,
    seed = 42,
    importance = "impurity"
  )

  # Predict probabilities
  probabilities <- predict(
    model,
    data = X_test
  )$predictions

  # Get predicted classes
  predicted <- class_names[
    max.col(as.matrix(probabilities))
  ]

  # Confusion matrix
  cm <- table(
    Actual = factor(y_test, levels = class_names),
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

  balanced_accuracy <- mean(recall[is.finite(recall)])
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
      Actual = y_test,
      Predicted = predicted
    ),
    paste0(
      "Data_Processed/extra_trees_",
      tolower(task_name), "_predictions.csv"
    ),
    row.names = FALSE
  )

  # Return results
  return(data.frame(
    Task = task_name,
    Accuracy = round(accuracy * 100, 2),
    Balanced_Accuracy = round(balanced_accuracy * 100, 2),
    Macro_F1 = round(macro_f1 * 100, 2)
  ))
}

# 3. Threat Classification
threat_results <- train_extra_trees(
  "Threat",
  "Threat_Level",
  "Data_Processed/Common_Splits/Threat_train.csv",
  "Data_Processed/Common_Splits/Threat_test.csv"
)

# 4. Species Classification
species_results <- train_extra_trees(
  "Species",
  "Species",
  "Data_Processed/Common_Splits/Species_train.csv",
  "Data_Processed/Common_Splits/Species_test.csv"
)

# 5. Final Results
final_results <- rbind(
  threat_results,
  species_results
)

cat("\n========== EXTRA TREES RESULTS ==========\n")
print(final_results)

write.csv(
  final_results,
  "Data_Processed/extra_trees_final_results.csv",
  row.names = FALSE
)
