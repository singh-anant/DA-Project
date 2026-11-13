
# Stage 2: SVM Feature Selection

library(e1071)

# ------------------------------------------------
# 1. Load datasets
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

# Select 33 features
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
  stop("Expected 33 features.")
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

# Remove non-finite values
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

# Convert target to factor
train_data[[threat_col]] <- factor(
  train_data[[threat_col]],
  levels = class_names
)

test_data[[threat_col]] <- factor(
  test_data[[threat_col]],
  levels = class_names
)

# ------------------------------------------------
# 2. Create stratified 5-fold CV
# ------------------------------------------------

make_folds <- function(y, k = 5) {

  set.seed(2026)

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

fold_id <- make_folds(
  train_data[[threat_col]]
)

# ------------------------------------------------
# 3. Feature ranking function
# ------------------------------------------------

rank_features <- function(data, feature_list, target, number) {

  p_values <- sapply(feature_list, function(feature) {

    result <- tryCatch(
      kruskal.test(
        data[[feature]] ~ data[[target]]
      )$p.value,
      error = function(e) 1
    )

    if (!is.finite(result)) {
      result <- 1
    }

    return(result)
  })

  ranked_features <- names(
    sort(p_values)
  )

  return(
    head(ranked_features, number)
  )
}

# ------------------------------------------------
# 4. Calculate inverse-frequency weights
# ------------------------------------------------

calculate_weights <- function(y) {

  counts <- table(
    factor(y, levels = class_names)
  )

  weights <- sum(counts) /
    (length(class_names) * counts)

  names(weights) <- class_names

  return(weights)
}

# ------------------------------------------------
# 5. Evaluation function
# ------------------------------------------------

evaluate_model <- function(actual, predicted) {

  cm <- table(
    Actual = factor(actual, levels = class_names),
    Predicted = factor(predicted, levels = class_names)
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
      TP + FP == 0, 0, TP / (TP + FP)
    )

    recall[i] <- ifelse(
      TP + FN == 0, 0, TP / (TP + FN)
    )

    f1[i] <- ifelse(
      precision[i] + recall[i] == 0,
      0,
      2 * precision[i] * recall[i] /
        (precision[i] + recall[i])
    )
  }

  return(data.frame(
    Accuracy = accuracy,
    Balanced_Accuracy = mean(recall),
    Macro_F1 = mean(f1),
    Normal_Precision = precision[2],
    Normal_Recall = recall[2],
    Normal_F1 = f1[2]
  ))
}

# ------------------------------------------------
# 6. Test different feature counts
# ------------------------------------------------

feature_counts <- c(5, 8, 10, 15, 20, 25, 33)

cv_results <- data.frame()

for (number in feature_counts) {

  cat("\nTesting", number, "features...\n")

  oof_predictions <- rep(
    NA_character_,
    nrow(train_data)
  )

  for (fold in 1:5) {

    train_indices <- which(fold_id != fold)
    validation_indices <- which(fold_id == fold)

    fold_train <- train_data[train_indices, ]
    fold_valid <- train_data[validation_indices, ]

    # Select features using training fold only
    selected_features <- rank_features(
      fold_train,
      features,
      threat_col,
      number
    )

    # Calculate training-fold class weights
    weights <- calculate_weights(
      fold_train[[threat_col]]
    )

    # Train SVM
    model <- svm(
      x = fold_train[, selected_features],
      y = fold_train[[threat_col]],
      kernel = "radial",
      cost = 100,
      gamma = 0.01,
      class.weights = weights,
      scale = TRUE
    )

    # Predict validation data
    predictions <- predict(
      model,
      newdata = fold_valid[, selected_features]
    )

    oof_predictions[validation_indices] <- as.character(
      predictions
    )
  }

  # Evaluate out-of-fold predictions
  metrics <- evaluate_model(
    train_data[[threat_col]],
    oof_predictions
  )

  cv_results <- rbind(
    cv_results,
    data.frame(
      Number_of_Features = number,
      metrics
    )
  )
}

# ------------------------------------------------
# 7. Find best feature counts
# ------------------------------------------------

best_accuracy <- cv_results[
  order(
    -cv_results$Accuracy,
    -cv_results$Macro_F1
  ),
][1, ]

best_macro <- cv_results[
  order(
    -cv_results$Macro_F1,
    -cv_results$Balanced_Accuracy
  ),
][1, ]

cat("\n========== CV RESULTS ==========\n")
print(cv_results)

cat("\n========== BEST BY ACCURACY ==========\n")
print(best_accuracy)

cat("\n========== BEST BY MACRO-F1 ==========\n")
print(best_macro)

# ------------------------------------------------
# 8. Train selected models on full training data
# ------------------------------------------------

train_and_test <- function(number, model_name) {

  # Select features using full training data
  selected_features <- rank_features(
    train_data,
    features,
    threat_col,
    number
  )

  weights <- calculate_weights(
    train_data[[threat_col]]
  )

  model <- svm(
    x = train_data[, selected_features],
    y = train_data[[threat_col]],
    kernel = "radial",
    cost = 100,
    gamma = 0.01,
    class.weights = weights,
    scale = TRUE
  )

  predictions <- predict(
    model,
    newdata = test_data[, selected_features]
  )

  metrics <- evaluate_model(
    test_data[[threat_col]],
    predictions
  )

  return(list(
    metrics = data.frame(
      Model = model_name,
      Number_of_Features = number,
      metrics
    ),
    features = selected_features,
    confusion = table(
      Actual = factor(
        test_data[[threat_col]],
        levels = class_names
      ),
      Predicted = factor(
        predictions,
        levels = class_names
      )
    )
  ))
}

baseline_test <- train_and_test(
  33,
  "Existing 33 Feature SVM"
)

accuracy_test <- train_and_test(
  best_accuracy$Number_of_Features,
  "Best CV Accuracy SVM"
)

macro_test <- train_and_test(
  best_macro$Number_of_Features,
  "Best CV Macro-F1 SVM"
)

test_comparison <- rbind(
  baseline_test$metrics,
  accuracy_test$metrics,
  macro_test$metrics
)

# ------------------------------------------------
# 9. Save results
# ------------------------------------------------

output_dir <- "Data_Processed/SVM_Feature_Selection"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

write.csv(
  cv_results,
  paste0(output_dir, "/cv_results.csv"),
  row.names = FALSE
)

write.csv(
  test_comparison,
  paste0(output_dir, "/test_comparison.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(Feature = accuracy_test$features),
  paste0(output_dir, "/best_accuracy_features.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(Feature = macro_test$features),
  paste0(output_dir, "/best_macro_f1_features.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(accuracy_test$confusion),
  paste0(output_dir, "/accuracy_confusion_matrix.csv")
)

write.csv(
  as.data.frame.matrix(macro_test$confusion),
  paste0(output_dir, "/macro_f1_confusion_matrix.csv")
)

# ------------------------------------------------
# 10. Display final results
# ------------------------------------------------

cat("\n========== FINAL TEST COMPARISON ==========\n")
print(test_comparison)

cat("\nResults saved in:", output_dir, "\n")
