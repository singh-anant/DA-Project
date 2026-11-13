
# ============================================================
# SPECIES-INDEPENDENT THREAT LEVEL EVALUATION
# RBF SVM
# ============================================================

library(e1071)

# 1. PROJECT PATHS

DATA_FILE <- "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv"

OUTPUT_DIR <- "Data_Processed/Species_Independent_Evaluation"

dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)


# 2. SELECT 33 AUDIO FEATURES

feature_cols <- c(
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


# 3. LOAD AND PREPARE DATA

data <- read.csv(DATA_FILE, check.names = FALSE)

# Remove Unknown threat labels
data <- data[data$Threat_Level %in%
               c("Dangerous", "Weak", "Normal"), ]

# Remove rows with missing or invalid values
required_cols <- c("Species", "File_Name", "Threat_Level",
                   feature_cols)

data <- data[complete.cases(data[, required_cols]), ]

data <- data[
  apply(data[, feature_cols], 1,
        function(x) all(is.finite(as.numeric(x)))),
]

data$Threat_Level <- factor(
  data$Threat_Level,
  levels = c("Dangerous", "Normal", "Weak")
)

cat("Total usable records:", nrow(data), "\n")
cat("Unique species:", length(unique(data$Species)), "\n")


# 4. SPLIT BY SPECIES

# All recordings of one species must stay together

set.seed(2026)

species_names <- unique(data$Species)

class_levels <- levels(data$Threat_Level)

# Select approximately 20% of species for testing
test_species_count <- ceiling(length(species_names) * 0.20)

# Try different splits until all classes appear in both sets
valid_split <- FALSE

for (attempt in 1:5000) {

  selected_species <- sample(
    species_names,
    test_species_count
  )

  train_data <- data[
    !data$Species %in% selected_species, ]

  test_data <- data[
    data$Species %in% selected_species, ]

  if (all(class_levels %in% unique(train_data$Threat_Level)) &&
      all(class_levels %in% unique(test_data$Threat_Level))) {

    valid_split <- TRUE
    break
  }
}

if (!valid_split) {
  stop("Could not create a split containing all classes.")
}

cat("\nTraining records:", nrow(train_data), "\n")
cat("Testing records:", nrow(test_data), "\n")

cat("Training species:",
    length(unique(train_data$Species)), "\n")

cat("Testing species:",
    length(unique(test_data$Species)), "\n")


# 5. VERIFY NO SPECIES OVERLAP

common_species <- intersect(
  unique(train_data$Species),
  unique(test_data$Species)
)

cat("\nSpecies overlap:", length(common_species), "\n")

if (length(common_species) > 0) {
  stop("Species leakage detected!")
}


# 6. CHECK CLASS DISTRIBUTION

cat("\nTraining class distribution:\n")
print(table(train_data$Threat_Level))

cat("\nTesting class distribution:\n")
print(table(test_data$Threat_Level))


# 7. PREPARE FEATURES

x_train <- train_data[, feature_cols]
x_test <- test_data[, feature_cols]

y_train <- train_data$Threat_Level
y_test <- factor(
  test_data$Threat_Level,
  levels = class_levels
)


# 8. CALCULATE CLASS WEIGHTS

class_counts <- table(
  factor(y_train, levels = class_levels)
)

class_weights <- length(y_train) /
  (length(class_levels) * as.numeric(class_counts))

names(class_weights) <- class_levels

print(class_weights)


# 9. TRAIN SVM

# Reuse previously selected Threat SVM parameters

model <- e1071::svm(
  x = x_train,
  y = y_train,
  type = "C-classification",
  kernel = "radial",
  cost = 100,
  gamma = 0.01,
  class.weights = class_weights,
  scale = TRUE
)


# 10. MAKE PREDICTIONS

predictions <- predict(model, x_test)


# 11. CONFUSION MATRIX

cm <- table(
  Actual = y_test,
  Predicted = factor(predictions, levels = class_levels)
)

cat("\nConfusion Matrix:\n")
print(cm)


# 12. CALCULATE METRICS

correct <- diag(cm)

precision <- ifelse(
  colSums(cm) == 0,
  0,
  correct / colSums(cm)
)

recall <- ifelse(
  rowSums(cm) == 0,
  0,
  correct / rowSums(cm)
)

f1 <- ifelse(
  precision + recall == 0,
  0,
  2 * precision * recall / (precision + recall)
)

accuracy <- sum(correct) / sum(cm)

balanced_accuracy <- mean(recall)

macro_f1 <- mean(f1)


# 13. CLASS-WISE RESULTS

class_metrics <- data.frame(
  Class = class_levels,
  Support = as.numeric(rowSums(cm)),
  Precision = precision,
  Recall = recall,
  F1_Score = f1
)

cat("\nClass-wise Metrics:\n")
print(class_metrics)

cat("\nFINAL RESULTS\n")

cat("Accuracy:",
    round(accuracy * 100, 2), "%\n")

cat("Balanced Accuracy:",
    round(balanced_accuracy * 100, 2), "%\n")

cat("Macro F1:",
    round(macro_f1 * 100, 2), "%\n")


# 14. SAVE RESULTS

write.csv(
  as.data.frame.matrix(cm),
  file.path(OUTPUT_DIR, "confusion_matrix.csv")
)

write.csv(
  class_metrics,
  file.path(OUTPUT_DIR, "class_metrics.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(
    File_Name = test_data$File_Name,
    Species = test_data$Species,
    Actual = y_test,
    Predicted = predictions
  ),
  file.path(OUTPUT_DIR, "predictions.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(
    Metric = c("Accuracy", "Balanced_Accuracy", "Macro_F1"),
    Score = c(accuracy, balanced_accuracy, macro_f1)
  ),
  file.path(OUTPUT_DIR, "final_summary.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(Species = unique(train_data$Species)),
  file.path(OUTPUT_DIR, "training_species.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(Species = unique(test_data$Species)),
  file.path(OUTPUT_DIR, "testing_species.csv"),
  row.names = FALSE
)

cat("\nSpecies-independent evaluation completed!\n")
cat("Results saved in:", OUTPUT_DIR, "\n")
