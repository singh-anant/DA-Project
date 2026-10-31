
# =====================================================================
# Stage 7: Support Vector Machine (SVM) Classification
#
# Trains separate Species and Threat_Level SVM classifiers.
#
# Uses:
# - 33 richer audio features
# - Previously created common 80:20 train-test splits
# - RBF kernel
# - 5-fold cross-validation for Cost and Gamma tuning
# - Class weights for imbalanced datasets
#
# IMPORTANT:
# No new train-test split is created in this script.
# =====================================================================
library(e1071)
library(readr)
library(dplyr)



# ------------------------------------------------------------
# Safe CSV saving function
# ------------------------------------------------------------

safe_write_csv <- function(data, file_path) {

  # Convert data into a regular data frame
  data <- as.data.frame(data)

  # Process each column
  for (i in seq_along(data)) {

    column <- data[[i]]

    # Convert matrix columns into text
    if (is.matrix(column)) {

      data[[i]] <- apply(
        column,
        1,
        function(x) paste(x, collapse = "; ")
      )

    }

    # Convert list columns into text
    else if (is.list(column)) {

      data[[i]] <- vapply(
        column,
        function(x) {
          paste(as.character(unlist(x)), collapse = "; ")
        },
        character(1)
      )

    }
  }

  # IMPORTANT: Call readr's original write_csv function.
  # Do NOT call safe_write_csv here.
  readr::write_csv(data, file_path)
}



# ------------------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------------------

split_dir <- "Data_Processed/Common_Splits"
output_dir <- "Data_Processed"

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

features <- c(
  "Pitch_Centroid", "Energy", "ZCR",
  "Spectral_Bandwidth", "Spectral_RollOff85",
  "Spectral_Flatness", "Spectral_Entropy",
  paste0("MFCC_", 1:13, "_Mean"),
  paste0("MFCC_", 1:13, "_SD")
)

CV_FOLDS <- 5

COST_VALUES <- c(0.1, 1, 10, 100)
GAMMA_VALUES <- c(0.001, 0.01, 0.1, 1)

# ------------------------------------------------------------
# 2. Helper function to calculate class weights
# ------------------------------------------------------------

calculate_class_weights <- function(y) {

  class_counts <- table(y)

  weights <- sum(class_counts) /
    (length(class_counts) * class_counts)

  return(weights)
}

# ------------------------------------------------------------
# 3. Evaluation function
# ------------------------------------------------------------

evaluate_model <- function(actual, predicted, class_names, model_name) {

  actual <- factor(actual, levels = class_names)
  predicted <- factor(predicted, levels = class_names)

  cm <- table(
    Actual = actual,
    Predicted = predicted
  )

  # Calculate precision, recall and F1 for every class
  per_class <- do.call(rbind, lapply(class_names, function(class_name) {

    tp <- cm[class_name, class_name]
    fp <- sum(cm[, class_name]) - tp
    fn <- sum(cm[class_name, ]) - tp

    precision <- if ((tp + fp) == 0) {
      0
    } else {
      tp / (tp + fp)
    }

    recall <- if ((tp + fn) == 0) {
      0
    } else {
      tp / (tp + fn)
    }

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

  # Overall evaluation metrics
  summary <- data.frame(
    Model = model_name,
    Test_Records = sum(cm),
    Classes = length(class_names),
    Accuracy = sum(diag(cm)) / sum(cm),
    Balanced_Accuracy = mean(per_class$Recall),
    Macro_F1 = mean(per_class$F1_Score)
  )

  return(list(
    confusion_matrix = cm,
    per_class = per_class,
    summary = summary
  ))
}

# ------------------------------------------------------------
# 4. Function to load and validate common splits
# ------------------------------------------------------------

load_common_split <- function(task_name, target) {

  train_file <- file.path(
    split_dir,
    paste0(task_name, "_train.csv")
  )

  test_file <- file.path(
    split_dir,
    paste0(task_name, "_test.csv")
  )

  if (!file.exists(train_file) || !file.exists(test_file)) {
    stop("Common split files not found for: ", task_name)
  }

  train_df <- read_csv(train_file, show_col_types = FALSE)
  test_df <- read_csv(test_file, show_col_types = FALSE)

  required_columns <- c("File_Name", target, features)

  missing_train <- setdiff(required_columns, names(train_df))
  missing_test <- setdiff(required_columns, names(test_df))

  if (length(missing_train) > 0 || length(missing_test) > 0) {
    stop("Required columns are missing in the ", task_name, " split.")
  }

  # Ensure filenames are characters
  train_df$File_Name <- as.character(train_df$File_Name)
  test_df$File_Name <- as.character(test_df$File_Name)

  # Ensure target values are characters before factor conversion
  train_df[[target]] <- trimws(as.character(train_df[[target]]))
  test_df[[target]] <- trimws(as.character(test_df[[target]]))

  # Convert audio features into numeric format
  for (col in features) {
    train_df[[col]] <- suppressWarnings(as.numeric(train_df[[col]]))
    test_df[[col]] <- suppressWarnings(as.numeric(test_df[[col]]))
  }

  # Do not silently remove records because that would change
  # the common train-test split.
  for (col in features) {

    if (any(!is.finite(train_df[[col]]))) {
      stop("Invalid feature values found in training data: ", col)
    }

    if (any(!is.finite(test_df[[col]]))) {
      stop("Invalid feature values found in testing data: ", col)
    }
  }

  # Check filenames
  if (anyDuplicated(train_df$File_Name) ||
      anyDuplicated(test_df$File_Name)) {
    stop("Duplicate filenames found in ", task_name, " split.")
  }

  # Ensure training and testing filenames do not overlap
  overlap <- intersect(
    train_df$File_Name,
    test_df$File_Name
  )

  if (length(overlap) > 0) {
    stop("Train-test overlap found in ", task_name, " split.")
  }

  # Ensure target classes are identical in training and testing
  train_classes <- sort(unique(train_df[[target]]))
  test_classes <- sort(unique(test_df[[target]]))

  if (!setequal(train_classes, test_classes)) {
    stop("Training and testing classes do not match for ", task_name)
  }

  # Set consistent factor levels
  train_df[[target]] <- factor(
    train_df[[target]],
    levels = train_classes
  )

  test_df[[target]] <- factor(
    test_df[[target]],
    levels = train_classes
  )

  cat("\nLoaded common split:", task_name, "\n")
  cat("Training records:", nrow(train_df), "\n")
  cat("Testing records:", nrow(test_df), "\n")
  cat("Number of classes:", length(train_classes), "\n")
  cat("Number of features:", length(features), "\n")
  cat("Filename overlap:", length(overlap), "\n")

  return(list(
    train = train_df,
    test = test_df
  ))
}

# ------------------------------------------------------------
# 5. Train and evaluate one RBF SVM model
# ------------------------------------------------------------

train_svm <- function(train_df, test_df, target, model_name, seed) {

  y_train <- droplevels(train_df[[target]])

  class_names <- levels(y_train)

  # Check that every training class has enough records for CV
  if (any(table(y_train) < CV_FOLDS)) {
    stop("Insufficient training records for 5-fold CV: ", model_name)
  }

  # Calculate class weights using training data only
  class_weights <- calculate_class_weights(y_train)

  cat("\n========================================\n")
  cat("SVM model:", model_name, "\n")
  cat("Target:", target, "\n")
  cat("Training records:", nrow(train_df), "\n")
  cat("Testing records:", nrow(test_df), "\n")
  cat("Classes:", length(class_names), "\n")
  cat("Features:", length(features), "\n")
  cat("Kernel: RBF\n")
  cat("Cross-validation folds:", CV_FOLDS, "\n")

  cat("\nTraining class distribution:\n")
  print(table(y_train))

  cat("\nTuning Cost and Gamma...\n")

  # ----------------------------------------------------------
  # Hyperparameter tuning using training data only
  # ----------------------------------------------------------

  set.seed(seed)

  svm_tuning <- tune.svm(
    x = train_df[, features, drop = FALSE],
    y = y_train,

    kernel = "radial",

    cost = COST_VALUES,
    gamma = GAMMA_VALUES,

    scale = TRUE,

    class.weights = class_weights,

    tunecontrol = tune.control(
      sampling = "cross",
      cross = CV_FOLDS
    )
  )

  best_model <- svm_tuning$best.model

  cat("\nBest parameters:\n")
  print(svm_tuning$best.parameters)

  cat(
    "Best CV error:",
    round(svm_tuning$best.performance, 4),
    "\n"
  )

  # ----------------------------------------------------------
  # Final prediction on the common test dataset
  # ----------------------------------------------------------

  predictions <- predict(
    best_model,
    newdata = test_df[, features, drop = FALSE]
  )

  evaluation <- evaluate_model(
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

  cat("\nConfusion Matrix:\n")
  print(evaluation$confusion_matrix)

  # ----------------------------------------------------------
  # Save trained model
  # ----------------------------------------------------------

  saveRDS(
    best_model,
    file.path(output_dir, paste0(model_name, "_svm_rbf.rds"))
  )

  

  # ----------------------------------------------------------
  # Save confusion matrix
  # ----------------------------------------------------------

  cm_df <- data.frame(
    Actual_Class = rownames(evaluation$confusion_matrix),
    as.data.frame.matrix(evaluation$confusion_matrix),
    check.names = FALSE
  )

  safe_write_csv(
    cm_df,
    file.path(output_dir, paste0(model_name, "_svm_confusion_matrix.csv"))
  )

  # ----------------------------------------------------------
  # Save class-wise metrics
  # ----------------------------------------------------------

  safe_write_csv(
    evaluation$per_class,
    file.path(output_dir, paste0(model_name, "_svm_class_metrics.csv"))
  )

  # ----------------------------------------------------------
  # Save predictions with original filenames
  # ----------------------------------------------------------

  prediction_df <- data.frame(
    File_Name = test_df$File_Name,
    Actual = as.character(test_df[[target]]),
    Predicted = as.character(predictions)
  )

  safe_write_csv(
    prediction_df,
    file.path(output_dir, paste0(model_name, "_svm_test_predictions.csv"))
  )

  # ----------------------------------------------------------
  # Save model summary
  # ----------------------------------------------------------

  summary_row <- evaluation$summary %>%
    mutate(
      Kernel = "Radial",
      Best_Cost = svm_tuning$best.parameters$cost[1],
      Best_Gamma = svm_tuning$best.parameters$gamma[1],
      CV_Folds = CV_FOLDS,
      CV_Error = svm_tuning$best.performance
    )

  return(list(summary = summary_row))
}

# ------------------------------------------------------------
# 6. Species classification using common split
# ------------------------------------------------------------

species_split <- load_common_split(
  task_name = "Species",
  target = "Species"
)

species_result <- train_svm(
  train_df = species_split$train,
  test_df = species_split$test,
  target = "Species",
  model_name = "species_richer_svm",
  seed = 42
)

# ------------------------------------------------------------
# 7. Threat Level classification using common split
# ------------------------------------------------------------

threat_split <- load_common_split(
  task_name = "Threat",
  target = "Threat_Level"
)

threat_result <- train_svm(
  train_df = threat_split$train,
  test_df = threat_split$test,
  target = "Threat_Level",
  model_name = "threat_richer_svm",
  seed = 43
)

# ------------------------------------------------------------
# 8. Save combined model summary
# ------------------------------------------------------------

model_summary <- bind_rows(
  species_result$summary,
  threat_result$summary
)

summary_file <- file.path(
  output_dir,
  "stage7_svm_richer_model_summary.csv"
)

safe_write_csv(model_summary, summary_file)

cat("\n========================================\n")
cat("STAGE 7 - SVM COMPLETE\n")
cat("========================================\n")

cat("\nFinal model summary:\n")
print(model_summary)

cat("\nSaved models:\n")
cat("- species_richer_svm_svm_rbf.rds\n")
cat("- threat_richer_svm_svm_rbf.rds\n")

cat("\nAll evaluation files saved in:", output_dir, "\n")
