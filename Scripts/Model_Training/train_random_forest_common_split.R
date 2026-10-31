# =====================================================================
# STAGE 7: RANDOM FOREST USING THE COMMON TRAIN-TEST SPLITS
# Species and Threat_Level classification with 33 richer audio features.
# No new split is created. The existing test files are used unchanged.
# =====================================================================

library(readr)
library(dplyr)
library(randomForest)

split_folder <- "Data_Processed/Common_Splits"
output_dir <- "Data_Processed"
NTREES <- 500
MTRY <- 6

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Select the same 33 richer audio features used by SVM and XGBoost.
features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)

cat("Total audio features:", length(features), "\n")

# Load the existing common split for the requested classification task.
load_common_split <- function(target) {

  # Threat_Level split files are named Threat_train/test.csv.
  split_name <- if (target == "Threat_Level") "Threat" else target

  train_path <- file.path(split_folder, paste0(split_name, "_train.csv"))
  test_path <- file.path(split_folder, paste0(split_name, "_test.csv"))

  if (!file.exists(train_path) || !file.exists(test_path)) {
    stop("Common split files not found for ", target,
         ". Check: ", split_folder)
  }

  train_df <- read_csv(train_path, show_col_types = FALSE)
  test_df <- read_csv(test_path, show_col_types = FALSE)

  required <- c("File_Name", target, features)
  missing_train <- setdiff(required, names(train_df))
  missing_test <- setdiff(required, names(test_df))

  if (length(missing_train) > 0 || length(missing_test) > 0) {
    stop(
      "Required columns missing.\nTraining: ",
      paste(missing_train, collapse = ", "),
      "\nTesting: ", paste(missing_test, collapse = ", ")
    )
  }

  # Ensure feature columns are numeric.
  for (feature in features) {
    train_df[[feature]] <- as.numeric(train_df[[feature]])
    test_df[[feature]] <- as.numeric(test_df[[feature]])
  }

  # Keep only valid rows. The original common split should already be clean.
  train_df <- train_df %>%
    filter(
      !is.na(File_Name), !is.na(.data[[target]]),
      if_all(all_of(features), ~ is.finite(.x))
    )

  test_df <- test_df %>%
    filter(
      !is.na(File_Name), !is.na(.data[[target]]),
      if_all(all_of(features), ~ is.finite(.x))
    )

  # Verify that no recording occurs in both sets.
  overlap <- intersect(train_df$File_Name, test_df$File_Name)
  cat("Filename overlap:", length(overlap), "\n")

  if (length(overlap) > 0) {
    stop("Data leakage detected: train/test filenames overlap.")
  }

  list(train = train_df, test = test_df)
}

# Equal prior weight for every class.
balanced_class_weights <- function(y) {
  y <- droplevels(as.factor(y))
  weights <- rep(1, nlevels(y))
  names(weights) <- levels(y)
  weights
}

# Calculate confusion matrix and classification metrics.
evaluate_model <- function(actual, predicted, class_names) {

  actual <- factor(actual, levels = class_names)
  predicted <- factor(predicted, levels = class_names)
  cm <- table(Actual = actual, Predicted = predicted)

  per_class <- do.call(rbind, lapply(class_names, function(class_name) {
    TP <- cm[class_name, class_name]
    FP <- sum(cm[, class_name]) - TP
    FN <- sum(cm[class_name, ]) - TP

    precision <- if ((TP + FP) == 0) 0 else TP / (TP + FP)
    recall <- if ((TP + FN) == 0) 0 else TP / (TP + FN)
    f1 <- if ((2 * TP + FP + FN) == 0) {
      0
    } else {
      2 * TP / (2 * TP + FP + FN)
    }

    data.frame(
      Class = class_name,
      Precision = precision,
      Recall = recall,
      F1_Score = f1
    )
  }))

  list(
    confusion_matrix = cm,
    per_class = per_class,
    accuracy = sum(diag(cm)) / sum(cm),
    balanced_accuracy = mean(per_class$Recall),
    macro_f1 = mean(per_class$F1_Score)
  )
}

# Train and evaluate one target using the existing common split.
train_and_evaluate <- function(target, model_name, seed_value) {

  cat("\n========================================\n")
  cat("RANDOM FOREST:", model_name, "\n")
  cat("Target:", target, "\n")
  cat("========================================\n")

  split_data <- load_common_split(target)
  train_df <- split_data$train
  test_df <- split_data$test

  train_df[[target]] <- droplevels(as.factor(train_df[[target]]))
  class_names <- levels(train_df[[target]])
  test_labels <- as.character(test_df[[target]])

  # Confirm every test class is represented in training.
  unseen <- setdiff(unique(test_labels), class_names)
  if (length(unseen) > 0) {
    stop("Test contains classes absent from training: ",
         paste(unseen, collapse = ", "))
  }

  cat("Training records:", nrow(train_df), "\n")
  cat("Testing records:", nrow(test_df), "\n")
  cat("Number of classes:", length(class_names), "\n")
  cat("Number of features:", length(features), "\n")

  cat("\nTraining class distribution:\n")
  print(table(train_df[[target]]))
  cat("\nTesting class distribution:\n")
  print(table(test_labels))

  weights <- balanced_class_weights(train_df[[target]])
  cat("\nClass prior weights:\n")
  print(weights)

  set.seed(seed_value)
  model <- randomForest(
    x = train_df[, features, drop = FALSE],
    y = train_df[[target]],
    ntree = NTREES,
    mtry = MTRY,
    classwt = weights,
    importance = TRUE
  )

  predictions <- predict(
    model,
    newdata = test_df[, features, drop = FALSE]
  )

  evaluation <- evaluate_model(
    actual = test_labels,
    predicted = predictions,
    class_names = class_names
  )

  cat("\n========== TEST RESULTS ==========\n")
  cat("Accuracy:", round(evaluation$accuracy * 100, 2), "%\n")
  cat("Balanced Accuracy:",
      round(evaluation$balanced_accuracy * 100, 2), "%\n")
  cat("Macro-F1:", round(evaluation$macro_f1 * 100, 2), "%\n")
  cat("\nConfusion Matrix (rows = actual, columns = predicted):\n")
  print(evaluation$confusion_matrix)

  # Save the model and its prediction metadata.
  saveRDS(
    list(
      model = model,
      features = features,
      class_names = class_names,
      target = target,
      ntree = NTREES,
      mtry = MTRY,
      class_weights = weights
    ),
    file.path(output_dir,
              paste0(model_name, "_random_forest_common_split.rds"))
  )

  # Save confusion matrix.
  cm_df <- data.frame(
    Actual_Class = rownames(evaluation$confusion_matrix),
    as.data.frame.matrix(evaluation$confusion_matrix),
    check.names = FALSE
  )
  write_csv(
    cm_df,
    file.path(output_dir,
              paste0(model_name, "_rf_common_split_confusion_matrix.csv"))
  )

  # Save class-wise metrics.
  write_csv(
    evaluation$per_class,
    file.path(output_dir,
              paste0(model_name, "_rf_common_split_class_metrics.csv"))
  )

  # Save feature importance.
  importance_values <- randomForest::importance(model)
  importance_df <- data.frame(
    Feature = rownames(importance_values),
    importance_values,
    check.names = FALSE,
    row.names = NULL
  )
  write_csv(
    importance_df,
    file.path(output_dir,
              paste0(model_name, "_rf_common_split_feature_importance.csv"))
  )

  # Save predictions for the exact common test records.
  prediction_df <- data.frame(
    File_Name = test_df$File_Name,
    Actual = test_labels,
    Predicted = as.character(predictions)
  )
  write_csv(
    prediction_df,
    file.path(output_dir,
              paste0(model_name, "_rf_common_split_predictions.csv"))
  )

  data.frame(
    Algorithm = "Random Forest",
    Target = target,
    Train_Records = nrow(train_df),
    Test_Records = nrow(test_df),
    Classes = length(class_names),
    Accuracy = round(evaluation$accuracy * 100, 2),
    Balanced_Accuracy = round(evaluation$balanced_accuracy * 100, 2),
    Macro_F1 = round(evaluation$macro_f1 * 100, 2)
  )
}

# Run Species classification.
species_results <- train_and_evaluate(
  target = "Species",
  model_name = "species",
  seed_value = 42
)

# Run Threat Level classification.
threat_results <- train_and_evaluate(
  target = "Threat_Level",
  model_name = "threat",
  seed_value = 43
)

# Save combined results.
final_results <- bind_rows(species_results, threat_results)

cat("\n========================================\n")
cat("RANDOM FOREST - COMMON SPLIT RESULTS\n")
cat("========================================\n")
print(final_results)

write_csv(
  final_results,
  file.path(output_dir, "random_forest_common_split_summary.csv")
)

cat("\nSTAGE 7 - RANDOM FOREST COMPLETE\n")
cat("Results saved in:", output_dir, "\n")
