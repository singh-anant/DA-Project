# =====================================================================
# Stage 7: K-Nearest Neighbors (KNN) Classification
#
# Trains two KNN classifiers using the richer audio feature dataset:
#   1. Species classification
#   2. Threat_Level classification
#
# Method:
#   - 33 numerical audio features
#   - Separate stratified 80:20 train-test split for each target
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
output_dir <- "Data_Processed"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file,
       "\nMake sure the richer final dataset exists.")
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
  stop("Input file is missing required columns: ",
       paste(missing_columns, collapse = ", "))
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
  stop("Duplicate File_Name values found.")
}

cat("Usable rows:", nrow(df), "\n")
cat("Audio predictors:", length(features), "\n")

# --------------------------- 3. Helper functions ----------------------

# Create a stratified train-test split.
stratified_split <- function(data, target, train_fraction = 0.80, seed = 42) {

  y <- droplevels(as.factor(data[[target]]))
  class_counts <- table(y)

  if (any(class_counts < 2)) {
    too_small <- names(class_counts)[class_counts < 2]
    stop("Classes with fewer than 2 recordings in ", target, ": ",
         paste(too_small, collapse = ", "))
  }

  set.seed(seed)

  train_indices <- unlist(
    lapply(names(class_counts), function(class_name) {
      indices <- which(y == class_name)
      n_train <- round(length(indices) * train_fraction)
      n_train <- max(1, min(n_train, length(indices) - 1))
      sample(indices, size = n_train)
    }),
    use.names = FALSE
  )

  list(
    train = data[train_indices, , drop = FALSE],
    test = data[-train_indices, , drop = FALSE]
  )
}

# Custom cross-validation metric: Macro-F1.
# This prevents the K value from being selected using accuracy alone.
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
    Model = model_name,
    Test_Records = sum(cm),
    Classes = length(class_names),
    Accuracy = sum(diag(cm)) / sum(cm),
    Balanced_Accuracy = mean(per_class$Recall),
    Macro_F1 = mean(per_class$F1_Score)
  )

  list(
    confusion_matrix = cm,
    per_class = per_class,
    summary = summary
  )
}

# --------------------------- 4. Train and evaluate KNN -----------------

train_knn <- function(data, target, model_name, seed) {

  data[[target]] <- droplevels(as.factor(data[[target]]))

  # Create a separate stratified split for this classification task.
  split <- stratified_split(
    data,
    target,
    train_fraction = 0.80,
    seed = seed
  )

  train_df <- split$train
  test_df <- split$test

  y_train <- droplevels(as.factor(train_df[[target]]))
  class_names <- levels(y_train)

  # Ensure each class has enough examples for 5-fold CV.
  if (any(table(y_train) < CV_FOLDS)) {
    stop("At least one training class has fewer than ", CV_FOLDS,
         " records for ", model_name, ".")
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

  # Stratified 5-fold CV on training data only.
  set.seed(seed)
  cv_control <- trainControl(
    method = "cv",
    number = CV_FOLDS,
    summaryFunction = macro_f1_summary,
    savePredictions = "final",
    allowParallel = FALSE
  )

  # KNN is distance-based, so center and scale the features.
  # caret estimates preprocessing from each training fold.
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
  cat("Best cross-validation Macro-F1:",
      round(max(knn_tuning$results$MacroF1), 4), "\n")

  # Predict the held-out test set.
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

  list(
    model = knn_tuning,
    summary = summary_row
  )
}

# --------------------------- 5. Species classification -----------------

# Exclude unknown identities and species with fewer than 20 recordings.
species_candidates <- df %>%
  filter(
    !is.na(Species),
    Species != "",
    tolower(Species) != "identity unknown"
  ) %>%
  count(Species, name = "Recordings") %>%
  filter(Recordings >= MIN_SPECIES_RECORDINGS) %>%
  select(Species)

species_df <- df %>%
  filter(
    !is.na(Species),
    Species != "",
    tolower(Species) != "identity unknown"
  ) %>%
  semi_join(species_candidates, by = "Species") %>%
  select(File_Name, Species, all_of(features)) %>%
  mutate(Species = droplevels(as.factor(Species)))

if (nrow(species_df) == 0 || nlevels(species_df$Species) < 2) {
  stop("Not enough eligible species data for KNN classification.")
}

species_result <- train_knn(
  data = species_df,
  target = "Species",
  model_name = "species_richer_knn",
  seed = 42
)

# --------------------------- 6. Threat classification ------------------

# Exclude Unknown/unresolved labels from supervised training.
threat_df <- df %>%
  filter(Threat_Level %in% c("Dangerous", "Weak", "Normal")) %>%
  select(File_Name, Threat_Level, all_of(features)) %>%
  mutate(
    Threat_Level = factor(
      Threat_Level,
      levels = c("Dangerous", "Weak", "Normal")
    )
  ) %>%
  filter(!is.na(Threat_Level)) %>%
  droplevels()

if (nrow(threat_df) == 0 || nlevels(threat_df$Threat_Level) < 2) {
  stop("Not enough eligible threat-labelled data for KNN classification.")
}

threat_result <- train_knn(
  data = threat_df,
  target = "Threat_Level",
  model_name = "threat_richer_knn",
  seed = 43
)

# --------------------------- 7. Save combined summary ------------------

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
cat("STAGE 7 - KNN COMPLETE\n")
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
