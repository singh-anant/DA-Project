
# Compare Species Identity Model vs Audio Feature SVM

library(e1071)

# Read the existing common random split
train_data <- read.csv(
  "Data_Processed/Common_Splits/Threat_train.csv"
)

test_data <- read.csv(
  "Data_Processed/Common_Splits/Threat_test.csv"
)

# Identify threat column
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
  stop("Expected 33 audio features.")
}

# Keep required columns and remove invalid rows
required_columns <- c("Species", threat_col, features)

train_data <- train_data[
  complete.cases(train_data[, required_columns]),
]

test_data <- test_data[
  complete.cases(test_data[, required_columns]),
]

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

# Set threat classes
class_names <- c("Dangerous", "Normal", "Weak")

train_data[[threat_col]] <- factor(
  train_data[[threat_col]],
  levels = class_names
)

test_data[[threat_col]] <- factor(
  test_data[[threat_col]],
  levels = class_names
)

# ------------------------------------------------
# Evaluation function
# ------------------------------------------------

evaluate_model <- function(actual, predicted, model_name) {

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

  metrics <- data.frame(
    Model = model_name,
    Accuracy = accuracy,
    Balanced_Accuracy = mean(recall),
    Macro_F1 = mean(f1)
  )

  class_metrics <- data.frame(
    Model = model_name,
    Class = class_names,
    Precision = precision,
    Recall = recall,
    F1 = f1,
    Support = rowSums(cm)
  )

  list(
    metrics = metrics,
    class_metrics = class_metrics,
    confusion = cm
  )
}

# ------------------------------------------------
# MODEL 1: Species Identity Only
# ------------------------------------------------

# Find the most frequent threat class in training data
global_majority <- names(
  which.max(table(train_data[[threat_col]]))
)

# Create a species-to-threat mapping
species_groups <- split(
  train_data[[threat_col]],
  train_data$Species
)

species_lookup <- sapply(
  species_groups,
  function(x) names(which.max(table(x)))
)

# Predict using species identity only
species_predictions <- sapply(
  test_data$Species,
  function(species) {

    if (species %in% names(species_lookup)) {
      species_lookup[[species]]
    } else {
      global_majority
    }
  }
)

species_predictions <- factor(
  species_predictions,
  levels = class_names
)

species_result <- evaluate_model(
  test_data[[threat_col]],
  species_predictions,
  "Species Identity Only"
)

# Calculate species coverage
known_species <- sum(
  test_data$Species %in% names(species_lookup)
)

species_coverage <- known_species / nrow(test_data)

# ------------------------------------------------
# MODEL 2: Audio Feature SVM
# ------------------------------------------------

# Calculate inverse-frequency class weights
class_counts <- table(
  factor(train_data[[threat_col]], levels = class_names)
)

class_weights <- sum(class_counts) /
  (length(class_names) * class_counts)

names(class_weights) <- class_names

# Train SVM using audio features
audio_model <- svm(
  x = train_data[, features],
  y = train_data[[threat_col]],
  kernel = "radial",
  cost = 100,
  gamma = 0.01,
  class.weights = class_weights,
  scale = TRUE
)

# Predict test records
audio_predictions <- predict(
  audio_model,
  newdata = test_data[, features]
)

audio_result <- evaluate_model(
  test_data[[threat_col]],
  audio_predictions,
  "Audio Feature SVM"
)

# ------------------------------------------------
# Compare both models
# ------------------------------------------------

comparison <- rbind(
  species_result$metrics,
  audio_result$metrics
)

class_comparison <- rbind(
  species_result$class_metrics,
  audio_result$class_metrics
)

# ------------------------------------------------
# Save results
# ------------------------------------------------

output_dir <- "Data_Processed/Species_vs_Audio_Analysis"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

write.csv(
  comparison,
  paste0(output_dir, "/model_comparison.csv"),
  row.names = FALSE
)

write.csv(
  class_comparison,
  paste0(output_dir, "/class_comparison.csv"),
  row.names = FALSE
)

write.csv(
  as.data.frame.matrix(species_result$confusion),
  paste0(output_dir, "/species_confusion_matrix.csv")
)

write.csv(
  as.data.frame.matrix(audio_result$confusion),
  paste0(output_dir, "/audio_confusion_matrix.csv")
)

# Print results
cat("\n========== MODEL COMPARISON ==========\n")
print(comparison)

cat("\n========== CLASS-WISE COMPARISON ==========\n")
print(class_comparison)

cat("\nSpecies coverage in test set:",
    round(species_coverage * 100, 2), "%\n")

cat("\nTraining species:",
    length(unique(train_data$Species)), "\n")

cat("Testing species:",
    length(unique(test_data$Species)), "\n")

cat("\nResults saved in:", output_dir, "\n")
