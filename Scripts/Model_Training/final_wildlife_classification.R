# ============================================================
# FINAL WILDLIFE AUDIO CLASSIFICATION SYSTEM
# Species Classification + Threat Level Classification
#
# Run from the DA-Project root directory:
# Rscript Scripts/final_wildlife_classification.R
#
# Output:
# Models/final_wildlife_classification.rds
# ============================================================

if (!requireNamespace("e1071", quietly = TRUE)) {
  install.packages("e1071")
}

library(e1071)

set.seed(42)

# ------------------------------------------------------------
# 1. PATHS AND MODEL CONFIGURATION
# ------------------------------------------------------------

split_dir <- "Data_Processed/Common_Splits"
model_dir <- "Models"
model_path <- file.path(model_dir, "final_wildlife_classification.rds")

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

species_config <- list(
  kernel = "radial",
  cost = 10,
  gamma = 0.01,
  class_weighting = "None"
)

threat_config <- list(
  kernel = "radial",
  cost = 100,
  gamma = 0.01,
  class_weighting = "Inverse frequency"
)

threat_classes <- c("Dangerous", "Normal", "Weak")

# ------------------------------------------------------------
# 2. DATA LOADING AND VALIDATION
# ------------------------------------------------------------

load_dataset <- function(file_name, target, class_levels = NULL) {

  file_path <- file.path(split_dir, file_name)

  if (!file.exists(file_path)) {
    stop("Dataset not found: ", file_path)
  }

  data <- read.csv(file_path, check.names = FALSE)

  required_columns <- c(features, target)

  missing_columns <- setdiff(required_columns, names(data))

  if (length(missing_columns) > 0) {
    stop(
      "Missing required columns in ", file_name, ": ",
      paste(missing_columns, collapse = ", ")
    )
  }

  data <- data[
    !is.na(data[[target]]) &
      trimws(as.character(data[[target]])) != "",
    ,
    drop = FALSE
  ]

  if (is.null(class_levels)) {
    class_levels <- sort(unique(as.character(data[[target]])))
  }

  labels <- as.character(data[[target]])

  if (any(!labels %in% class_levels)) {
    stop("Unexpected class label found in ", file_name)
  }

  x <- as.data.frame(lapply(
    data[, features, drop = FALSE],
    function(column) suppressWarnings(as.numeric(as.character(column)))
  ))

  x <- as.matrix(x)
  storage.mode(x) <- "double"

  if (any(!is.finite(x))) {
    stop(
      "Missing or non-finite feature values found in ", file_name,
      ". Check the feature dataset before training."
    )
  }

  list(
    x = x,
    y = factor(labels, levels = class_levels),
    class_levels = class_levels
  )
}


# ------------------------------------------------------------
# 3. EVALUATION METRICS
# ------------------------------------------------------------

evaluate_model <- function(actual, predicted, class_levels, task_name) {

  cm <- table(
    Actual = factor(actual, levels = class_levels),
    Predicted = factor(predicted, levels = class_levels)
  )

  true_positive <- diag(cm)

  precision <- ifelse(
    colSums(cm) > 0,
    true_positive / pmax(colSums(cm), 1),
    0
  )

  recall <- ifelse(
    rowSums(cm) > 0,
    true_positive / pmax(rowSums(cm), 1),
    0
  )

  f1 <- ifelse(
    precision + recall > 0,
    2 * precision * recall / (precision + recall),
    0
  )

  accuracy <- sum(true_positive) / sum(cm)
  balanced_accuracy <- mean(recall)
  macro_f1 <- mean(f1)

  cat("\n", task_name, "\n", sep = "")
  cat(paste(rep("-", 58), collapse = ""), "\n", sep = "")
  cat(sprintf("Test samples       : %d\n", sum(cm)))
  cat(sprintf("Accuracy           : %.2f%%\n", accuracy * 100))
  cat(sprintf("Balanced accuracy  : %.2f%%\n", balanced_accuracy * 100))
  cat(sprintf("Macro-F1           : %.2f%%\n", macro_f1 * 100))

  #cat("\nConfusion Matrix (Actual x Predicted)\n")
  #print(cm)

  class_metrics <- data.frame(
    Class = class_levels,
    Support = as.integer(rowSums(cm)),
    Precision = round(as.numeric(precision), 3),
    Recall = round(as.numeric(recall), 3),
    F1_Score = round(as.numeric(f1), 3)
  )

  #cat("\nClass-wise Performance\n")
  #print(class_metrics, row.names = FALSE)

  data.frame(
    Classification = task_name,
    Test_Samples = sum(cm),
    Accuracy = accuracy * 100,
    Balanced_Accuracy = balanced_accuracy * 100,
    Macro_F1 = macro_f1 * 100
  )
}


# ------------------------------------------------------------
# 4. LOAD EXISTING TRAIN AND TEST SPLITS
# ------------------------------------------------------------

species_train <- load_dataset("Species_train.csv", "Species")
species_test <- load_dataset(
  "Species_test.csv",
  "Species",
  species_train$class_levels
)

threat_train <- load_dataset(
  "Threat_train.csv",
  "Threat_Level",
  threat_classes
)

threat_test <- load_dataset(
  "Threat_test.csv",
  "Threat_Level",
  threat_classes
)

cat("\nFINAL WILDLIFE AUDIO CLASSIFICATION\n")
cat("==================================\n")
cat("Features used       : 33 audio features\n")
cat("Species training    :", nrow(species_train$x), "\n")
cat("Species test        :", nrow(species_test$x), "\n")
cat("Threat training     :", nrow(threat_train$x), "\n")
cat("Threat test         :", nrow(threat_test$x), "\n")


# ------------------------------------------------------------
# 5. TRAIN AND EVALUATE SPECIES MODEL
# ------------------------------------------------------------

cat("\n[1/2] Evaluating Species Classification...\n")

species_eval_model <- e1071::svm(
  x = species_train$x,
  y = species_train$y,
  type = "C-classification",
  kernel = species_config$kernel,
  cost = species_config$cost,
  gamma = species_config$gamma,
  scale = TRUE
)

species_test_predictions <- predict(
  species_eval_model,
  newdata = species_test$x
)

species_results <- evaluate_model(
  species_test$y,
  species_test_predictions,
  species_train$class_levels,
  "SPECIES CLASSIFICATION"
)


# ------------------------------------------------------------
# 6. TRAIN AND EVALUATE THREAT LEVEL MODEL
# ------------------------------------------------------------

cat("\n[2/2] Evaluating Threat Level Classification...\n")

calculate_threat_weights <- function(labels) {

  counts <- table(factor(labels, levels = threat_classes))

  if (any(counts == 0)) {
    stop("At least one threat class has no training samples.")
  }

  weights <- length(labels) /
    (length(threat_classes) * as.numeric(counts))

  names(weights) <- threat_classes
  weights
}

threat_weights_train <- calculate_threat_weights(threat_train$y)

threat_eval_model <- e1071::svm(
  x = threat_train$x,
  y = threat_train$y,
  type = "C-classification",
  kernel = threat_config$kernel,
  cost = threat_config$cost,
  gamma = threat_config$gamma,
  class.weights = threat_weights_train,
  scale = TRUE
)

threat_test_predictions <- predict(
  threat_eval_model,
  newdata = threat_test$x
)

threat_results <- evaluate_model(
  threat_test$y,
  threat_test_predictions,
  threat_classes,
  "THREAT LEVEL CLASSIFICATION"
)


# ------------------------------------------------------------
# 7. COMBINED TEST EVALUATION SUMMARY
# ------------------------------------------------------------

final_summary <- rbind(species_results, threat_results)

cat("\n\nFINAL TEST PERFORMANCE SUMMARY\n")
cat(paste(rep("=", 58), collapse = ""), "\n", sep = "")
print(
  transform(
    final_summary,
    Accuracy = round(Accuracy, 2),
    Balanced_Accuracy = round(Balanced_Accuracy, 2),
    Macro_F1 = round(Macro_F1, 2)
  ),
  row.names = FALSE
)


# ------------------------------------------------------------
# 8. TRAIN DEPLOYMENT MODELS USING ALL AVAILABLE LABELED DATA
# ------------------------------------------------------------
# Test metrics above are calculated using train-only fitted models.
# After evaluation, refit each deployment model on its complete
# train + test split to use all available labeled recordings.

cat("\nPreparing final deployment models...\n")

species_all_x <- rbind(species_train$x, species_test$x)
species_all_y <- factor(
  c(as.character(species_train$y), as.character(species_test$y)),
  levels = species_train$class_levels
)

threat_all_x <- rbind(threat_train$x, threat_test$x)
threat_all_y <- factor(
  c(as.character(threat_train$y), as.character(threat_test$y)),
  levels = threat_classes
)

final_species_model <- e1071::svm(
  x = species_all_x,
  y = species_all_y,
  type = "C-classification",
  kernel = species_config$kernel,
  cost = species_config$cost,
  gamma = species_config$gamma,
  scale = TRUE
)

threat_weights_all <- calculate_threat_weights(threat_all_y)

final_threat_model <- e1071::svm(
  x = threat_all_x,
  y = threat_all_y,
  type = "C-classification",
  kernel = threat_config$kernel,
  cost = threat_config$cost,
  gamma = threat_config$gamma,
  class.weights = threat_weights_all,
  scale = TRUE
)


# ------------------------------------------------------------
# 9. SAVE BOTH MODELS IN ONE RDS FILE
# ------------------------------------------------------------

if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

final_model_bundle <- list(
  project = "Threat Level Detection of Wild Animals",
  model_version = "1.0",
  feature_names = features,
  species = list(
    model = final_species_model,
    classes = species_train$class_levels,
    configuration = species_config
  ),
  threat = list(
    model = final_threat_model,
    classes = threat_classes,
    configuration = threat_config,
    class_weights = threat_weights_all
  ),
  evaluation = final_summary,
  note = paste(
    "Deployment models were refitted using all available labeled",
    "train and test recordings after the holdout evaluation."
  )
)

saveRDS(final_model_bundle, model_path)

cat("\n", paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("MODEL TRAINING AND EVALUATION COMPLETED\n")
cat(paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("Saved model bundle: ", model_path, "\n", sep = "")
cat("Contains          : Species SVM + Threat Level SVM\n")
cat("Format            : RDS\n")


# ------------------------------------------------------------
# 9. PREDICTION FUNCTION FOR NEW FEATURE ROWS
# ------------------------------------------------------------

make_wildlife_predictor <- function(species_model, threat_model, feature_names) {

  function(new_features) {

    if (!is.data.frame(new_features)) {
      stop("new_features must be a data.frame.")
    }

    missing_features <- setdiff(feature_names, names(new_features))

    if (length(missing_features) > 0) {
      stop(
        "Missing audio features: ",
        paste(missing_features, collapse = ", ")
      )
    }

    x <- as.data.frame(lapply(
      new_features[, feature_names, drop = FALSE],
      function(column) suppressWarnings(as.numeric(as.character(column)))
    ))

    x <- as.matrix(x)
    storage.mode(x) <- "double"

    if (any(!is.finite(x))) {
      stop("New audio features contain missing or non-finite values.")
    }

    species_prediction <- predict(species_model, newdata = x)
    threat_prediction <- predict(threat_model, newdata = x)

    data.frame(
      Species = as.character(species_prediction),
      Threat_Level = as.character(threat_prediction)
    )
  }
}


# ------------------------------------------------------------
# 10. SAVE BOTH MODELS IN ONE RDS FILE
# ------------------------------------------------------------

if (!dir.exists(model_dir)) {
  dir.create(model_dir, recursive = TRUE)
}

wildlife_predictor <- make_wildlife_predictor(
  final_species_model,
  final_threat_model,
  features
)

final_model_bundle <- list(
  project = "Threat Level Detection of Wild Animals",
  model_version = "1.0",
  feature_names = features,
  species = list(
    model = final_species_model,
    classes = species_train$class_levels,
    configuration = species_config
  ),
  threat = list(
    model = final_threat_model,
    classes = threat_classes,
    configuration = threat_config,
    class_weights = threat_weights_all
  ),
  predict = wildlife_predictor,
  evaluation = final_summary,
  note = paste(
    "Deployment models were refitted using all available labeled",
    "train and test recordings after the holdout evaluation."
  )
)

saveRDS(final_model_bundle, model_path)

cat("\n", paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("MODEL TRAINING AND EVALUATION COMPLETED\n")
cat(paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("Saved model bundle: ", model_path, "\n", sep = "")
cat("Contains          : Species SVM + Threat Level SVM\n")
cat("Prediction method : final_model_bundle$predict(new_features)\n")
cat("Format            : RDS\n")


# ------------------------------------------------------------
# 11. EXAMPLE: PREDICT FROM EXTRACTED AUDIO FEATURES
# ------------------------------------------------------------
# The input must be a data.frame containing the 33 feature columns.
#
# final_model_bundle <- readRDS(
#   "Models/final_wildlife_classification.rds"
# )
#
# new_audio_features <- read.csv("one_audio_feature_row.csv")
#
# prediction <- final_model_bundle$predict(new_audio_features)
# print(prediction)
#
# This model bundle classifies extracted features. It does not
# extract features directly from raw MP3/WAV audio. Use the same
# audio feature extraction pipeline used to create the training data.
