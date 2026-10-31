# =====================================================================
# Stage 7: K-Nearest Neighbors (KNN) Classification
#
# Trains two KNN classifiers using the SAME common train-test split
# used by the SVM, XGBoost, and Random Forest models:
#   1. Species classification
#   2. Threat_Level classification
#
# Method:
#   - 33 numerical audio features
#   - Common train-test split (no new split is created)
#   - Center and scale features using training data
#   - 5-fold cross-validation to select the best K
#   - Select K using Macro-F1 during cross-validation
#
# Outputs use KNN-specific filenames and do not overwrite other models.
# =====================================================================

library(readr)
library(dplyr)
library(caret)

# --------------------------- 1. Paths and settings --------------------

input_file <- "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv"
split_dir <- "Data_Processed/Common_Splits"
output_dir <- "Data_Processed"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file)
}

# The 33 numerical audio features from feature_extraction_richer.R.
features <- c(
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

MIN_SPECIES_RECORDINGS <- 20
CV_FOLDS <- 5

# K values to compare during cross-validation.
K_VALUES <- c(3, 5, 7, 9, 11, 15, 21)

# --------------------------- 2. Read and validate data ----------------

df <- read_csv(input_file, show_col_types = FALSE)

required_columns <- c("File_Name", "Species", "Threat_Level", features)
missing_columns <- setdiff(required_columns, names(df))

if (length(missing_columns) > 0) {
  stop(
    "Input file is missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# Convert model features to numeric.
for (col in features) {
  df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
}

# Keep recordings with valid IDs and finite feature values.
df <- df %>%
  mutate(
    File_Name = as.character(File_Name),
    Species = trimws(as.character(Species)),
    Threat_Level = trimws(as.character(Threat_Level))
  ) %>%
  filter(
    !is.na(File_Name),
    File_Name != "",
    if_all(all_of(features), ~ is.finite(.x))
  )

if (nrow(df) == 0) {
  stop("No rows remain after feature-quality checks.")
}

if (anyDuplicated(df$File_Name)) {
  stop("Duplicate File_Name values found in the feature dataset.")
}

cat("Usable rows:", nrow(df), "\n")
cat("Audio predictors:", length(features), "\n")

# --------------------------- 3. Read common train-test splits ---------

read_split_files <- function(prefix) {
  train_file <- file.path(split_dir, paste0(prefix, "_train.csv"))
  test_file  <- file.path(split_dir, paste0(prefix, "_test.csv"))

  if (!file.exists(train_file) || !file.exists(test_file)) {
    stop(
      "Common split files not found for ", prefix, ".\n",
      "Expected:\n", train_file, "\n", test_file
    )
  }

  train_data <- read_csv(train_file, show_col_types = FALSE)
  test_data  <- read_csv(test_file, show_col_types = FALSE)

  list(train = train_data, test = test_data)
}

# Check split files and keep only the columns needed by KNN.
prepare_split <- function(split_data, target, model_name) {

  train_df <- split_data$train
  test_df <- split_data$test

  required <- c("File_Name", target, features)

  missing_train <- setdiff(required, names(train_df))
  missing_test <- setdiff(required, names(test_df))

  if (length(missing_train) > 0 || length(missing_test) > 0) {
    stop(
      "Required columns are missing from the ", model_name, " split files.\n",
      "Training missing: ", paste(missing_train, collapse = ", "), "\n",
      "Testing missing: ", paste(missing_test, collapse = ", ")
    )
  }

  train_df <- train_df %>%
    select(all_of(required)) %>%
    mutate(File_Name = as.character(File_Name))

  test_df <- test_df %>%
    select(all_of(required)) %>%
    mutate(File_Name = as.character(File_Name))

  # Ensure that training and testing contain no common recordings.
  overlap <- intersect(train_df$File_Name, test_df$File_Name)

  cat("\n", model_name, " filename overlap:", length(overlap), "\n")

  if (length(overlap) > 0) {
    stop("Data leakage detected: training and testing filenames overlap.")
  }

  # Convert feature columns to numeric and check for invalid values.
  for (col in features) {
    train_df[[col]] <- suppressWarnings(as.numeric(train_df[[col]]))
    test_df[[col]] <- suppressWarnings(as.numeric(test_df[[col]]))
  }

  if (any(!is.finite(as.matrix(train_df[, features]))) ||
      any(!is.finite(as.matrix(test_df[, features])))) {
    stop("Non-finite feature values found in ", model_name, " split.")
  }

  # Keep only the classes that are present in the training data.
  train_df[[target]] <- trimws(as.character(train_df[[target]]))
  test_df[[target]] <- trimws(as.character(test_df[[target]]))

  class_names <- sort(unique(train_df[[target]]))

  if (any(!test_df[[target]] %in% class_names)) {
    stop("Testing split contains a class absent from training for ", model_name)
  }

  train_df[[target]] <- factor(train_df[[target]], levels = class_names)
  test_df[[target]] <- factor(test_df[[target]], levels = class_names)

  list(train = train_df, test = test_df, classes = class_names)
}

# --------------------------- 4. Evaluation functions ------------------

# Custom cross-validation metric: Macro-F1.
# Each class contributes equally, regardless of its number of recordings.
macro_f1_summary <- function(data, lev = NULL, model = NULL) {

  actual <- factor(data$obs, levels = lev)
  predicted <- factor(data$pred, levels = lev)

  cm <- table(Actual = actual, Predicted = predicted)

  class_f1 <- sapply(lev, function(class_name) {
    tp <- cm[class_name, class_name]
    fp <- sum(cm[, class_name]) - tp
    fn <- sum(cm[class_name, ]) - tp

    denominator <- (2 * tp) + fp + fn

    if (denominator == 0) {
      return(0)
    }

    (2 * tp) / denominator
  })

  c(MacroF1 = mean(class_f1))
}

# Calculate final test metrics and confusion matrix.
evaluate_knn <- function(actual, predicted, class_names, model_name) {

  actual <- factor(actual, levels = class_names)
  predicted <- factor(predicted, levels = class_names)

  cm <- table(Actual = actual, Predicted = predicted)

  per_class <- do.call(
    rbind,
    lapply(class_names, function(class_name) {

      tp <- cm[class_name, class_name]
      fp <- sum(cm[, class_name]) - tp
      fn <- sum(cm[class_name, ]) - tp

      precision <- if ((tp + fp) == 0) 0 else tp / (tp + fp)
      recall <- if ((tp + fn) == 0) 0 else tp / (tp + fn)

      f1 <- if ((2 * tp + fp + fn) == 0) {
        0
      } else {
        (2 * tp) / (2 * tp + fp + fn)
      }

      data.frame(
        Model = model_name,
        Class = class_name,
        Precision = precision,
        Recall = recall,
        F1_Score = f1
      )
    })
  )

  summary <- data.frame(
    Algorithm = "KNN",
    Target = model_name,
    Test_Records = sum(cm),
    Classes = length(class_names),
    Accuracy = sum(diag(cm)) / sum(cm),
    # Multiclass balanced accuracy = mean class recall.
    Balanced_Accuracy = mean(per_class$Recall),
    Macro_F1 = mean(per_class$F1_Score)
  )

  list(
    confusion_matrix = cm,
    per_class = per_class,
    summary = summary
  )
}

# --------------------------- 5. Train and evaluate KNN ----------------

train_knn <- function(train_df, test_df, target, model_name, class_names, seed) {

  y_train <- droplevels(train_df[[target]])

  # Every class must have at least 5 training examples for 5-fold CV.
  if (any(table(y_train) < CV_FOLDS)) {
    stop(
      "At least one training class has fewer than ",
      CV_FOLDS, " records for ", model_name, "."
    )
  }

  cat("\n========================================\n")
  cat("KNN model:", model_name, "\n")
  cat("Target:", target, "\n")
  cat("Training rows:", nrow(train_df), "\n")
  cat("Testing rows:", nrow(test_df), "\n")
  cat("Classes:", length(class_names), "\n")
  cat("Features:", length(features), "\n")
  cat("Cross-validation folds:", CV_FOLDS, "\n")
  cat("K values:", paste(K_VALUES, collapse = ", "), "\n")
  cat("Tuning K using Macro-F1...\n")

  # Stratified 5-fold CV uses only the training split.
  set.seed(seed)

  cv_control <- trainControl(
    method = "cv",
    number = CV_FOLDS,
    summaryFunction = macro_f1_summary,
    savePredictions = "final",
    allowParallel = FALSE
  )

  # KNN is distance-based, so standardize the audio features.
  # caret estimates centering and scaling within each CV training fold.
  set.seed(seed)

  knn_tuning <- train(
    x = train_df[, features, drop = FALSE],
    y = y_train,
    method = "knn",
    metric = "MacroF1",
    maximize = TRUE,
    trControl = cv_control,
    preProcess = c("center", "scale"),
    tuneGrid = data.frame(k = K_VALUES)
  )

  best_k <- knn_tuning$bestTune$k

  cat("\nBest K:", best_k, "\n")
  cat(
    "Best cross-validation Macro-F1:",
    round(max(knn_tuning$results$MacroF1), 4), "\n"
  )

  # Predict the held-out common test split.
  predictions <- predict(
    knn_tuning,
    newdata = test_df[, features, drop = FALSE]
  )

  evaluation <- evaluate_knn(
    actual = test_df[[target]],
    predicted = predictions,
    class_names = class_names,
    model_name = model_name
  )

  cat("\nTest Accuracy:",
      round(evaluation$summary$Accuracy, 4), "\n")
  cat("Test Balanced Accuracy:",
      round(evaluation$summary$Balanced_Accuracy, 4), "\n")
  cat("Test Macro-F1:",
      round(evaluation$summary$Macro_F1, 4), "\n")

  cat("\nConfusion Matrix (rows = actual, columns = predicted):\n")
  print(evaluation$confusion_matrix)

  # Save the trained KNN model.
  saveRDS(
    knn_tuning,
    file.path(output_dir, paste0(model_name, "_knn.rds"))
  )

  # Save cross-validation results for all tested K values.
  write_csv(
    knn_tuning$results,
    file.path(output_dir, paste0(model_name, "_knn_tuning_results.csv"))
  )

  # Save confusion matrix.
  cm_df <- data.frame(
    Actual_Class = rownames(evaluation$confusion_matrix),
    as.data.frame.matrix(evaluation$confusion_matrix),
    check.names = FALSE
  )

  write_csv(
    cm_df,
    file.path(output_dir, paste0(model_name, "_knn_confusion_matrix.csv"))
  )

  # Save class-wise metrics.
  write_csv(
    evaluation$per_class,
    file.path(output_dir, paste0(model_name, "_knn_class_metrics.csv"))
  )

  # Save held-out predictions.
  prediction_df <- data.frame(
    File_Name = test_df$File_Name,
    Actual = as.character(test_df[[target]]),
    Predicted = as.character(predictions)
  )

  write_csv(
    prediction_df,
    file.path(output_dir, paste0(model_name, "_knn_test_predictions.csv"))
  )

  # Save a summary row including the selected K.
  summary_row <- evaluation$summary %>%
    mutate(
      Best_K = best_k,
      CV_Folds = CV_FOLDS,
      CV_Macro_F1 = max(knn_tuning$results$MacroF1)
    )

  list(model = knn_tuning, summary = summary_row)
}

# --------------------------- 6. Species classification ----------------

species_splits <- read_split_files("Species")

species_data <- prepare_split(
  split_data = species_splits,
  target = "Species",
  model_name = "Species"
)

species_result <- train_knn(
  train_df = species_data$train,
  test_df = species_data$test,
  target = "Species",
  model_name = "species_richer_knn",
  class_names = species_data$classes,
  seed = 42
)

# --------------------------- 7. Threat classification -----------------

threat_splits <- read_split_files("Threat")

threat_data <- prepare_split(
  split_data = threat_splits,
  target = "Threat_Level",
  model_name = "Threat"
)

threat_result <- train_knn(
  train_df = threat_data$train,
  test_df = threat_data$test,
  target = "Threat_Level",
  model_name = "threat_richer_knn",
  class_names = threat_data$classes,
  seed = 43
)

# --------------------------- 8. Save combined summary ------------------

model_summary <- bind_rows(
  species_result$summary,
  threat_result$summary
)

summary_file <- file.path(
  output_dir,
  "stage7_knn_richer_model_summary.csv"
)

write_csv(model_summary, summary_file)

cat("\n========================================\n")
cat("STAGE 7 - KNN COMMON SPLIT COMPLETE\n")
cat("\nKNN model summary:\n")
print(model_summary)

cat("\nSaved models:\n")
cat("- Data_Processed/species_richer_knn_knn.rds\n")
cat("- Data_Processed/threat_richer_knn_knn.rds\n")

cat("\nSaved evaluation files:\n")
cat("- *_knn_tuning_results.csv\n")
cat("- *_knn_confusion_matrix.csv\n")
cat("- *_knn_class_metrics.csv\n")
cat("- *_knn_test_predictions.csv\n")
cat("- Data_Processed/stage7_knn_richer_model_summary.csv\n")
