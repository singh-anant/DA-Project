# =====================================================================
# Stage 5: Random Forest Model Training (Richer Audio Features)
# Trains Species and Threat_Level classifiers using 33 audio features.
# =====================================================================

library(readr)
library(dplyr)
library(randomForest)

# 1. Paths and settings
input_file <- "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv"
output_dir <- "Data_Processed"

set.seed(42)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(input_file)) {
  stop("Input file not found: ", input_file,
       "\nRun the richer final dataset creation script first.")
}

# 33 richer numerical audio features:
# 3 original features + 4 additional spectral features
# + 13 MFCC means + 13 MFCC standard deviations.
features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)

MIN_SPECIES_RECORDINGS <- 20
NTREES <- 500

# 2. Read and validate data
df <- read_csv(input_file, show_col_types = FALSE)

required_columns <- c("File_Name", "Species", "Threat_Level", features)
missing_columns <- setdiff(required_columns, names(df))

if (length(missing_columns) > 0) {
  stop("The input file is missing required columns: ",
       paste(missing_columns, collapse = ", "))
}

for (col in features) {
  df[[col]] <- suppressWarnings(as.numeric(df[[col]]))
}

# Remove rows with missing/non-finite audio features.
# Metadata columns are not used as model predictors.
df <- df %>%
  mutate(
    File_Name = as.character(File_Name),
    Species = trimws(as.character(Species)),
    Threat_Level = trimws(as.character(Threat_Level))
  ) %>%
  filter(
    !is.na(File_Name), File_Name != "",
    if_all(all_of(features), ~ is.finite(.x))
  )

if (nrow(df) == 0) stop("No rows remain after feature-quality checks.")
if (anyDuplicated(df$File_Name)) {
  stop("Duplicate File_Name values found. Resolve duplicates before splitting.")
}

cat("Input rows with usable richer features:", nrow(df), "\n")
cat("Audio predictors:", length(features), "\n")

# 3. Helper functions

# Create a stratified split so each class is represented in train and test.
stratified_split <- function(data, target, train_fraction = 0.80, seed = 42) {
  y <- droplevels(as.factor(data[[target]]))
  class_counts <- table(y)

  if (any(class_counts < 2)) {
    too_small <- names(class_counts)[class_counts < 2]
    stop("Cannot split classes with fewer than 2 rows in ", target, ": ",
         paste(too_small, collapse = ", "))
  }

  set.seed(seed)
  train_indices <- unlist(lapply(names(class_counts), function(class_name) {
    indices <- which(y == class_name)
    n_train <- round(length(indices) * train_fraction)
    n_train <- max(1, min(n_train, length(indices) - 1))
    sample(indices, size = n_train)
  }), use.names = FALSE)

  list(
    train = data[train_indices, , drop = FALSE],
    test = data[-train_indices, , drop = FALSE]
  )
}

# randomForest classwt represents class prior weights.
# Equal weights assign equal prior importance to each class.
balanced_class_weights <- function(y) {
  y <- droplevels(as.factor(y))
  weights <- rep(1, nlevels(y))
  names(weights) <- levels(y)
  weights
}

# Calculate accuracy, per-class precision/recall/F1,
# macro-F1 and balanced accuracy.
evaluate_model <- function(actual, predicted, class_names, model_name) {
  actual <- factor(actual, levels = class_names)
  predicted <- factor(predicted, levels = class_names)
  cm <- table(Actual = actual, Predicted = predicted)

  per_class <- do.call(rbind, lapply(class_names, function(class_name) {
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
  }))

  accuracy <- if (sum(cm) == 0) NA_real_ else sum(diag(cm)) / sum(cm)
  macro_f1 <- mean(per_class$F1_Score)
  balanced_accuracy <- mean(per_class$Recall)

  summary <- data.frame(
    Model = model_name,
    Test_Records = sum(cm),
    Classes = length(class_names),
    Accuracy = accuracy,
    Balanced_Accuracy = balanced_accuracy,
    Macro_F1 = macro_f1
  )

  list(confusion_matrix = cm, per_class = per_class, summary = summary)
}

# Train, evaluate and save one Random Forest model.
train_and_evaluate <- function(data, target, model_name, seed) {
  data[[target]] <- droplevels(as.factor(data[[target]]))
  split <- stratified_split(data, target, train_fraction = 0.80, seed = seed)

  train_df <- split$train
  test_df <- split$test
  y_train <- droplevels(as.factor(train_df[[target]]))
  class_names <- levels(y_train)
  weights <- balanced_class_weights(y_train)

  unseen_test_classes <- setdiff(
    unique(as.character(test_df[[target]])), class_names
  )
  if (length(unseen_test_classes) > 0) {
    stop("Test contains classes absent from training for ", model_name, ".")
  }

  cat("\n========================================\n")
  cat("Model:", model_name, "\n")
  cat("Target:", target, "\n")
  cat("Usable rows:", nrow(data), "\n")
  cat("Training rows:", nrow(train_df), "\n")
  cat("Testing rows:", nrow(test_df), "\n")
  cat("Number of classes:", length(class_names), "\n")
  cat("Audio predictors:", length(features), "\n")
  cat("Class weighting: equal prior weight for each class\n")

  set.seed(seed)
  model <- randomForest(
    x = train_df[, features, drop = FALSE],
    y = y_train,
    ntree = NTREES,
    mtry = 6,
    classwt = weights,
    importance = TRUE
  )

  predictions <- predict(model, newdata = test_df[, features, drop = FALSE])
  evaluation <- evaluate_model(
    actual = test_df[[target]],
    predicted = predictions,
    class_names = class_names,
    model_name = model_name
  )

  cat("\nAccuracy:", round(evaluation$summary$Accuracy, 4), "\n")
  cat("Balanced accuracy:",
      round(evaluation$summary$Balanced_Accuracy, 4), "\n")
  cat("Macro-F1:", round(evaluation$summary$Macro_F1, 4), "\n")

  # Save trained model.
  saveRDS(model, file.path(output_dir,
                           paste0(model_name, "_random_forest.rds")))

  # Save confusion matrix.
  cm_df <- data.frame(
    Actual_Class = rownames(evaluation$confusion_matrix),
    as.data.frame.matrix(evaluation$confusion_matrix),
    check.names = FALSE
  )
  write_csv(cm_df, file.path(output_dir,
                             paste0(model_name, "_confusion_matrix.csv")))

  # Save class-wise metrics.
  write_csv(evaluation$per_class,
            file.path(output_dir, paste0(model_name, "_class_metrics.csv")))

  # Save feature importance.
  importance_values <- importance(model)
  importance_df <- data.frame(
    Feature = rownames(importance_values),
    importance_values,
    check.names = FALSE,
    row.names = NULL
  )
  importance_column <- tail(colnames(importance_values), 1)
  importance_df <- importance_df %>%
    arrange(desc(.data[[importance_column]]))

  write_csv(importance_df, file.path(output_dir,
                                     paste0(model_name, "_feature_importance.csv")))

  # Save held-out test predictions.
  prediction_df <- data.frame(
    File_Name = test_df$File_Name,
    Actual = as.character(test_df[[target]]),
    Predicted = as.character(predictions)
  )
  write_csv(prediction_df, file.path(output_dir,
                                     paste0(model_name, "_test_predictions.csv")))

  list(
    model = model,
    summary = evaluation$summary,
    per_class = evaluation$per_class
  )
}

# 4. Species classifier
# Exclude unknown identities and species with fewer than 20 recordings.
species_candidates <- df %>%
  filter(
    !is.na(Species), Species != "",
    tolower(Species) != "identity unknown"
  ) %>%
  count(Species, name = "Recordings") %>%
  filter(Recordings >= MIN_SPECIES_RECORDINGS) %>%
  select(Species)

species_df <- df %>%
  filter(
    !is.na(Species), Species != "",
    tolower(Species) != "identity unknown"
  ) %>%
  semi_join(species_candidates, by = "Species") %>%
  select(File_Name, Species, all_of(features)) %>%
  mutate(Species = droplevels(as.factor(Species)))

if (nrow(species_df) == 0 || nlevels(species_df$Species) < 2) {
  stop("Not enough eligible species data to train the species classifier.")
}

species_result <- train_and_evaluate(
  data = species_df,
  target = "Species",
  model_name = "species",
  seed = 42
)

# 5. Threat classifier
# Exclude Unknown/unresolved labels from supervised training.
threat_df <- df %>%
  filter(Threat_Level %in% c("Dangerous", "Weak", "Normal")) %>%
  select(File_Name, Threat_Level, all_of(features)) %>%
  mutate(Threat_Level = factor(
    Threat_Level,
    levels = c("Dangerous", "Weak", "Normal")
  )) %>%
  filter(!is.na(Threat_Level)) %>%
  droplevels()

if (nrow(threat_df) == 0 || nlevels(threat_df$Threat_Level) < 2) {
  stop("Not enough eligible threat-labelled data to train the threat classifier.")
}

threat_result <- train_and_evaluate(
  data = threat_df,
  target = "Threat_Level",
  model_name = "threat",
  seed = 43
)

# 6. Save combined model summary.
model_summary <- bind_rows(species_result$summary, threat_result$summary)
write_csv(model_summary, file.path(output_dir, "stage5_model_summary.csv"))

cat("\n========================================\n")
cat("STAGE 5 COMPLETE\n")
cat("\nModel summary:\n")
print(model_summary)

cat("\nSaved model files:\n")
cat("- Data_Processed/species_random_forest.rds\n")
cat("- Data_Processed/threat_random_forest.rds\n")
cat("\nSaved evaluation files (for each model):\n")
cat("- *_confusion_matrix.csv\n")
cat("- *_class_metrics.csv\n")
cat("- *_feature_importance.csv\n")
cat("- *_test_predictions.csv\n")
cat("- Data_Processed/stage5_model_summary.csv\n")
