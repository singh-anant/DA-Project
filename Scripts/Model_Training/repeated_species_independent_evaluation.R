
# Repeated Species-Independent Threat Level Evaluation

# Load required library
library(e1071)

# Set random seed
set.seed(2026)

# Read dataset
data <- read.csv(
  "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv"
)

# Identify threat column
threat_candidates <- c("Threat_Level", "Threat", "Target_Label")
threat_col <- threat_candidates[threat_candidates %in% names(data)][1]

if (is.na(threat_col)) {
  stop("Threat column not found. Check the dataset column names.")
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

mfcc_features <- grep("^MFCC", names(data), value = TRUE)

features <- c(basic_features, mfcc_features)

if (length(features) != 33) {
  stop("Expected 33 features, but found ", length(features))
}

# Remove Unknown labels and missing values
data <- data[
  data[[threat_col]] %in% c("Dangerous", "Normal", "Weak"),
]

data <- data[
  complete.cases(data[, c("Species", threat_col, features)]),
]

# Remove non-finite feature values
valid_rows <- apply(
  data[, features],
  1,
  function(x) all(is.finite(x))
)

data <- data[valid_rows, ]

# Convert target into factor
class_names <- c("Dangerous", "Normal", "Weak")

data[[threat_col]] <- factor(
  data[[threat_col]],
  levels = class_names
)

# Output directory
output_dir <- "Data_Processed/Repeated_Species_Independent_Evaluation"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# Store results
all_results <- data.frame()
all_class_metrics <- data.frame()
all_distributions <- data.frame()
all_confusion <- data.frame()

# Evaluation settings
number_of_repeats <- 10
test_percentage <- 0.20

# Calculate evaluation metrics
calculate_metrics <- function(actual, predicted) {

  cm <- table(
    Actual = factor(actual, levels = class_names),
    Predicted = factor(predicted, levels = class_names)
  )

  accuracy <- sum(diag(cm)) / sum(cm)

  precision <- numeric(length(class_names))
  recall <- numeric(length(class_names))
  f1 <- numeric(length(class_names))

  for (i in seq_along(class_names)) {

    TP <- cm[i, i]
    FP <- sum(cm[, i]) - TP
    FN <- sum(cm[i, ]) - TP

    precision[i] <- ifelse(TP + FP == 0, 0, TP / (TP + FP))
    recall[i] <- ifelse(TP + FN == 0, 0, TP / (TP + FN))

    f1[i] <- ifelse(
      precision[i] + recall[i] == 0,
      0,
      2 * precision[i] * recall[i] /
        (precision[i] + recall[i])
    )
  }

  balanced_accuracy <- mean(recall)
  macro_f1 <- mean(f1)

  class_metrics <- data.frame(
    Class = class_names,
    Precision = precision,
    Recall = recall,
    F1 = f1,
    Support = rowSums(cm)
  )

  list(
    accuracy = accuracy,
    balanced_accuracy = balanced_accuracy,
    macro_f1 = macro_f1,
    class_metrics = class_metrics,
    confusion = cm
  )
}

# Run repeated species-independent evaluation
for (repeat_no in 1:number_of_repeats) {

  cat("\nStarting Repeat:", repeat_no, "\n")

  # Try different random splits until all classes
  # are present in both training and testing sets
  split_found <- FALSE

  for (attempt in 1:5000) {

    set.seed(2026 + repeat_no * 10000 + attempt)

    all_species <- unique(data$Species)

    test_species <- sample(
      all_species,
      size = ceiling(length(all_species) * test_percentage)
    )

    train_data <- data[!data$Species %in% test_species, ]
    test_data <- data[data$Species %in% test_species, ]

    if (
      all(class_names %in% unique(train_data[[threat_col]])) &&
      all(class_names %in% unique(test_data[[threat_col]]))
    ) {
      split_found <- TRUE
      break
    }
  }

  if (!split_found) {
    stop(
      "Could not create a valid split for repeat ",
      repeat_no
    )
  }

  # Verify that species do not overlap
  overlap <- intersect(
    unique(train_data$Species),
    unique(test_data$Species)
  )

  if (length(overlap) > 0) {
    stop("Species overlap detected!")
  }

  # Calculate inverse-frequency class weights
  class_counts <- table(
    factor(train_data[[threat_col]], levels = class_names)
  )

  class_weights <- sum(class_counts) /
    (length(class_names) * class_counts)

  names(class_weights) <- class_names

  # Train SVM model
  svm_model <- svm(
    x = train_data[, features],
    y = train_data[[threat_col]],
    kernel = "radial",
    cost = 100,
    gamma = 0.01,
    class.weights = class_weights,
    scale = TRUE
  )

  # Predict on unseen species
  predictions <- predict(
    svm_model,
    newdata = test_data[, features]
  )

  # Calculate metrics
  results <- calculate_metrics(
    test_data[[threat_col]],
    predictions
  )

  # Store overall results
  all_results <- rbind(
    all_results,
    data.frame(
      Repeat = repeat_no,
      Train_Records = nrow(train_data),
      Test_Records = nrow(test_data),
      Train_Species = length(unique(train_data$Species)),
      Test_Species = length(unique(test_data$Species)),
      Accuracy = results$accuracy,
      Balanced_Accuracy = results$balanced_accuracy,
      Macro_F1 = results$macro_f1
    )
  )

  # Store class-wise metrics
  class_output <- results$class_metrics
  class_output$Repeat <- repeat_no

  all_class_metrics <- rbind(
    all_class_metrics,
    class_output
  )

  # Store class distributions
  for (class_name in class_names) {

    all_distributions <- rbind(
      all_distributions,
      data.frame(
        Repeat = repeat_no,
        Dataset = "Training",
        Class = class_name,
        Count = sum(train_data[[threat_col]] == class_name)
      ),
      data.frame(
        Repeat = repeat_no,
        Dataset = "Testing",
        Class = class_name,
        Count = sum(test_data[[threat_col]] == class_name)
      )
    )
  }

  # Store confusion matrix
  confusion_output <- as.data.frame(
    as.table(results$confusion)
  )

  confusion_output$Repeat <- repeat_no

  all_confusion <- rbind(
    all_confusion,
    confusion_output
  )

  cat("Accuracy:", round(results$accuracy * 100, 2), "%\n")
  cat("Balanced Accuracy:",
      round(results$balanced_accuracy * 100, 2), "%\n")
  cat("Macro-F1:", round(results$macro_f1 * 100, 2), "%\n")
}

# Calculate mean and standard deviation
metric_names <- c(
  "Accuracy",
  "Balanced_Accuracy",
  "Macro_F1"
)

summary_results <- data.frame(
  Metric = metric_names,
  Mean = sapply(all_results[, metric_names], mean),
  SD = sapply(all_results[, metric_names], sd)
)

# Save all results
write.csv(
  all_results,
  paste0(output_dir, "/repeat_results.csv"),
  row.names = FALSE
)

write.csv(
  summary_results,
  paste0(output_dir, "/summary_mean_sd.csv"),
  row.names = FALSE
)

write.csv(
  all_class_metrics,
  paste0(output_dir, "/class_metrics.csv"),
  row.names = FALSE
)

write.csv(
  all_distributions,
  paste0(output_dir, "/class_distributions.csv"),
  row.names = FALSE
)

write.csv(
  all_confusion,
  paste0(output_dir, "/confusion_matrices.csv"),
  row.names = FALSE
)

# Calculate Normal class average performance
normal_results <- all_class_metrics[
  all_class_metrics$Class == "Normal",
]

normal_summary <- data.frame(
  Metric = c("Precision", "Recall", "F1"),
  Mean = c(
    mean(normal_results$Precision),
    mean(normal_results$Recall),
    mean(normal_results$F1)
  ),
  SD = c(
    sd(normal_results$Precision),
    sd(normal_results$Recall),
    sd(normal_results$F1)
  )
)

write.csv(
  normal_summary,
  paste0(output_dir, "/normal_class_summary.csv"),
  row.names = FALSE
)

# Display final results
cat("\n========== FINAL SUMMARY ==========\n")
print(summary_results)

cat("\n========== NORMAL CLASS SUMMARY ==========\n")
print(normal_summary)

cat("\nAll results saved in:", output_dir, "\n")
