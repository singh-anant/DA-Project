
# Stage 1: Optimize SVM for Threat Level Classification

library(e1071)

# ------------------------------------------------
# 1. Load common training and testing datasets
# ------------------------------------------------

train_data <- read.csv(
  "Data_Processed/Common_Splits/Threat_train.csv"
)

test_data <- read.csv(
  "Data_Processed/Common_Splits/Threat_test.csv"
)

# Identify target column
threat_candidates <- c("Threat_Level", "Threat", "Target_Label")

threat_col <- threat_candidates[
  threat_candidates %in% names(train_data)
][1]

if (is.na(threat_col)) {
  stop("Threat column not found.")
}

# Select 33 audio features
basic_features <- c(
  "Pitch_Centroid",
  "Energy",
  "ZCR",
  "Spectral_Bandwidth",
  "Spectral_RollOff85",
  "Spectral_Flatness",
  "Spectral_Entropy"
)

mfcc_features <- grep("^MFCC", names(train_data), value = TRUE)

features <- c(basic_features, mfcc_features)

if (length(features) != 33) {
  stop("Expected 33 features. Found: ", length(features))
}

# Threat classes
class_names <- c("Dangerous", "Normal", "Weak")

# Remove missing values
required_columns <- c(threat_col, features)

train_data <- train_data[
  complete.cases(train_data[, required_columns]),
]

test_data <- test_data[
  complete.cases(test_data[, required_columns]),
]

# Remove non-finite feature values
train_valid <- apply(
  train_data[, features],
  1,
  function(x) all(is.finite(x))
)

test_valid <- apply(
  test_data[, features],
  1,
  function(x) all(is.finite(x))
)

train_data <- train_data[train_valid, ]
test_data <- test_data[test_valid, ]

# Convert target into factor
train_data[[threat_col]] <- factor(
  train_data[[threat_col]],
  levels = class_names
)

test_data[[threat_col]] <- factor(
  test_data[[threat_col]],
  levels = class_names
)

# ------------------------------------------------
# 2. Create stratified 5-fold cross-validation
# ------------------------------------------------

make_stratified_folds <- function(y, k = 5, seed = 2026) {

  set.seed(seed)

  fold_id <- integer(length(y))

  for (class_name in levels(y)) {

    indices <- which(y == class_name)

    indices <- sample(indices)

    fold_id[indices] <- rep(
      1:k,
      length.out = length(indices)
    )
  }

  return(fold_id)
}

fold_id <- make_stratified_folds(
  train_data[[threat_col]],
  k = 5
)

cat("Training records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")
cat("Number of folds: 5\n")

# ------------------------------------------------
# 3. Function to calculate class weights
# ------------------------------------------------

calculate_weights <- function(y, method) {

  if (method == "none") {
    return(NULL)
  }

  counts <- table(
    factor(y, levels = class_names)
  )

  if (any(counts == 0)) {
    stop("A training fold is missing a threat class.")
  }

  total <- sum(counts)
  number_classes <- length(class_names)

  # Inverse-frequency weights
  weights <- total / (number_classes * counts)

  if (method == "sqrt_inverse") {
    weights <- sqrt(weights)
  }

  names(weights) <- class_names

  return(weights)
}

# ------------------------------------------------
# 4. Evaluation function
# ------------------------------------------------

evaluate_model <- function(actual, predicted) {

  actual <- factor(
    actual,
    levels = class_names
  )

  predicted <- factor(
    predicted,
    levels = class_names
  )

  cm <- table(
    Actual = actual,
    Predicted = predicted
  )

  accuracy <- sum(diag(cm)) / sum(cm)

  precision <- numeric(3)
  recall <- numeric(3)
  f1 <- numeric(3)

  for (i in 1:3) {

    TP <- cm[i, i]
    FP <- sum(cm[, i]) - TP
    FN <- sum(cm[i, ]) - TP

    precision[i] <- ifelse(
      TP + FP == 0,
      0,
      TP / (TP + FP)
    )

    recall[i] <- ifelse(
      TP + FN == 0,
      0,
      TP / (TP + FN)
    )

    f1[i] <- ifelse(
      precision[i] + recall[i] == 0,
      0,
      2 * precision[i] * recall[i] /
        (precision[i] + recall[i])
    )
  }

  metrics <- data.frame(
    Accuracy = accuracy,
    Balanced_Accuracy = mean(recall),
    Macro_F1 = mean(f1),
    Normal_Precision = precision[2],
    Normal_Recall = recall[2],
    Normal_F1 = f1[2]
  )

  return(list(
    metrics = metrics,
    confusion = cm
  ))
}

# ------------------------------------------------
# 5. Define SVM parameter combinations
# ------------------------------------------------

C_values <- c(0.1, 1, 10, 100, 1000)

gamma_values <- c(
  0.0001,
  0.001,
  0.01,
  0.1,
  1
)

weight_methods <- c(
  "none",
  "sqrt_inverse",
  "inverse"
)

# 75 configurations:
# 3 weighting methods x 5 C values x 5 Gamma values

parameter_grid <- expand.grid(
  Weighting = weight_methods,
  C = C_values,
  Gamma = gamma_values,
  stringsAsFactors = FALSE
)

# ------------------------------------------------
# 6. Perform 5-fold cross-validation
# ------------------------------------------------

cv_results <- list()

total_models <- nrow(parameter_grid)

for (config_no in 1:total_models) {

  config <- parameter_grid[config_no, ]

  # Store out-of-fold predictions
  oof_predictions <- rep(
    NA_character_,
    nrow(train_data)
  )

  # Run all five folds
  for (fold in 1:5) {

    train_indices <- which(fold_id != fold)
    validation_indices <- which(fold_id == fold)

    fold_train <- train_data[train_indices, ]
    fold_valid <- train_data[validation_indices, ]

    # Calculate weights using only fold training data
    weights <- calculate_weights(
      fold_train[[threat_col]],
      config$Weighting
    )

    # Train SVM
    model <- svm(
      x = fold_train[, features],
      y = fold_train[[threat_col]],
      kernel = "radial",
      cost = config$C,
      gamma = config$Gamma,
      class.weights = weights,
      scale = TRUE
    )

    # Predict validation data
    predictions <- predict(
      model,
      newdata = fold_valid[, features]
    )

    oof_predictions[validation_indices] <- as.character(
      predictions
    )
  }

  # Evaluate all out-of-fold predictions together
  evaluation <- evaluate_model(
    train_data[[threat_col]],
    oof_predictions
  )

  result <- data.frame(
    Weighting = config$Weighting,
    C = config$C,
    Gamma = config$Gamma,
    evaluation$metrics
  )

  cv_results[[config_no]] <- result

  cat(
    "Completed configuration",
    config_no,
    "of",
    total_models,
    "\n"
  )
}

# Combine results
cv_results <- do.call(rbind, cv_results)

# Sort by Accuracy, then Macro-F1, then Balanced Accuracy
cv_results <- cv_results[
  order(
    -cv_results$Accuracy,
    -cv_results$Macro_F1,
    -cv_results$Balanced_Accuracy
  ),
]

# Also identify the best model by Macro-F1
best_macro <- cv_results[
  order(
    -cv_results$Macro_F1,
    -cv_results$Balanced_Accuracy,
    -cv_results$Accuracy
  ),
][1, ]

# Best model by Accuracy
best_accuracy <- cv_results[1, ]

# Existing optimized SVM configuration
baseline <- cv_results[
  cv_results$Weighting == "inverse" &
    cv_results$C == 100 &
    cv_results$Gamma == 0.01,
]

# ------------------------------------------------
# 7. Train and test selected configurations
# ------------------------------------------------

train_and_test <- function(config, model_name) {

  weights <- calculate_weights(
    train_data[[threat_col]],
    config$Weighting
  )

  model <- svm(
    x = train_data[, features],
    y = train_data[[threat_col]],
    kernel = "radial",
    cost = config$C,
    gamma = config$Gamma,
    class.weights = weights,
    scale = TRUE
  )

  predictions <- predict(
    model,
    newdata = test_data[, features]
  )

  evaluation <- evaluate_model(
    test_data[[threat_col]],
    predictions
  )

  metrics <- data.frame(
    Model = model_name,
    Weighting = config$Weighting,
    C = config$C,
    Gamma = config$Gamma,
    evaluation$metrics
  )

  list(
    metrics = metrics,
    class_metrics = data.frame(
      Model = model_name,
      Class = class_names,
      Precision = sapply(1:3, function(i) {
        cm <- evaluation$confusion
        TP <- cm[i, i]
        FP <- sum(cm[, i]) - TP
        ifelse(TP + FP == 0, 0, TP / (TP + FP))
      }),
      Recall = sapply(1:3, function(i) {
        cm <- evaluation$confusion
        TP <- cm[i, i]
        FN <- sum(cm[i, ]) - TP
        ifelse(TP + FN == 0, 0, TP / (TP + FN))
      })
    ),
    confusion = evaluation$confusion
  )
}

# Test existing baseline and best accuracy model
baseline_test <- train_and_test(
  baseline,
  "Existing SVM"
)

best_test <- train_and_test(
  best_accuracy,
  "Best CV Accuracy SVM"
)

macro_test <- train_and_test(
  best_macro,
  "Best CV Macro-F1 SVM"
)

test_comparison <- rbind(
  baseline_test$metrics,
  best_test$metrics,
  macro_test$metrics
)



cat("\n========== FINAL THREE-MODEL COMPARISON ==========\n")
print(test_comparison)

# ------------------------------------------------
# 8. Save all results
# ------------------------------------------------

# Create output directory
output_dir <- "Data_Processed/SVM_Stage1_Optimization"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# Save test comparison
write.csv(
  test_comparison,
  paste0(output_dir, "/test_comparison.csv"),
  row.names = FALSE
)

# Save best Accuracy model confusion matrix
write.csv(
  as.data.frame.matrix(best_test$confusion),
  paste0(output_dir, "/best_confusion_matrix.csv")
)

# Save best Macro-F1 model confusion matrix
write.csv(
  as.data.frame.matrix(macro_test$confusion),
  paste0(output_dir, "/macro_f1_confusion_matrix.csv")
)

# ------------------------------------------------
# 9. Display final results
# ------------------------------------------------

cat("\n========== TOP 10 CV CONFIGURATIONS ==========\n")
print(head(cv_results, 10))

cat("\n========== BEST CONFIGURATION BY ACCURACY ==========\n")
print(best_accuracy)

cat("\n========== BEST CONFIGURATION BY MACRO-F1 ==========\n")
print(best_macro)

cat("\n========== HELD-OUT TEST COMPARISON ==========\n")
print(test_comparison)

cat("\nResults saved in:", output_dir, "\n")
