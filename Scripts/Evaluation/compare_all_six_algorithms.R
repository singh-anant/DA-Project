# =====================================================================
# STAGE 8: CONSOLIDATED COMPARISON OF SIX CLASSIFICATION ALGORITHMS
#
# Algorithms:
#   1. SVM
#   2. XGBoost
#   3. Random Forest
#   4. KNN
#   5. Gaussian Naive Bayes
#   6. Decision Tree
#
# Tasks:
#   1. Species classification
#   2. Threat Level classification
#
# This script reads the saved test predictions from Stage 7.
# It does NOT retrain models or create a new train-test split.
#
# Balanced Accuracy is calculated consistently as:
#   Mean of the recall values of all classes.
#
# Macro-F1 is the mean of the class-wise F1 scores.
# =====================================================================

library(readr)
library(dplyr)
library(stringr)
library(tidyr)

# --------------------------- 1. Paths -------------------------------

data_dir <- "Data_Processed"
split_dir <- file.path(data_dir, "Common_Splits")
output_dir <- file.path(data_dir, "Final_Model_Comparison")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------- 2. Read common test splits --------------

species_test_file <- file.path(split_dir, "Species_test.csv")
threat_test_file <- file.path(split_dir, "Threat_test.csv")

if (!file.exists(species_test_file) || !file.exists(threat_test_file)) {
  stop("Common test split files are missing from Data_Processed/Common_Splits.")
}

species_test <- read_csv(species_test_file, show_col_types = FALSE)
threat_test <- read_csv(threat_test_file, show_col_types = FALSE)

if (!all(c("File_Name", "Species") %in% names(species_test))) {
  stop("Species_test.csv must contain File_Name and Species columns.")
}

if (!all(c("File_Name", "Threat_Level") %in% names(threat_test))) {
  stop("Threat_test.csv must contain File_Name and Threat_Level columns.")
}

species_test <- species_test %>%
  transmute(
    File_Name = as.character(File_Name),
    Actual = trimws(as.character(Species))
  )

threat_test <- threat_test %>%
  transmute(
    File_Name = as.character(File_Name),
    Actual = trimws(as.character(Threat_Level))
  )

# --------------------------- 3. Find prediction files ----------------

# Search recursively in Data_Processed.
all_csv_files <- list.files(
  data_dir,
  pattern = "\\.csv$",
  recursive = TRUE,
  full.names = TRUE
)

# Keep CSV files that look like saved test-prediction files.
prediction_files <- all_csv_files[
  str_detect(tolower(basename(all_csv_files)), "prediction")
]

if (length(prediction_files) == 0) {
  stop(
    "No prediction CSV files found.\n",
    "Expected files containing columns: File_Name, Actual, Predicted."
  )
}

# --------------------------- 4. Identify algorithm and task ----------

identify_algorithm <- function(file_name) {

  name <- tolower(file_name)

  if (str_detect(name, "decision.?tree")) {
    return("Decision Tree")
  }

  if (str_detect(name, "xgboost|xgb")) {
    return("XGBoost")
  }

  # Match both descriptive names (random_forest) and abbreviations
  # surrounded by underscores, as in species_rf_common_split_predictions.csv.
  if (str_detect(name, "random.?forest|(^|_)rf(_|\\.)")) {
    return("Random Forest")
  }

  if (str_detect(name, "gaussian.?nb|naive.?bayes|\\bnb\\b")) {
    return("Gaussian NB")
  }

  if (str_detect(name, "knn|k.?nearest")) {
    return("KNN")
  }

  if (str_detect(name, "svm")) {
    return("SVM")
  }

  NA_character_
}

identify_task <- function(file_name) {

  name <- tolower(file_name)

  if (str_detect(name, "species")) {
    return("Species")
  }

  if (str_detect(name, "threat")) {
    return("Threat")
  }

  NA_character_
}

# --------------------------- 5. Metric calculation -------------------

calculate_metrics <- function(prediction_data, test_data,
                               algorithm, task) {

  # Check required columns.
  required <- c("File_Name", "Actual", "Predicted")

  if (!all(required %in% names(prediction_data))) {
    stop("Prediction file does not contain all required columns.")
  }

  prediction_data <- prediction_data %>%
    transmute(
      File_Name = as.character(File_Name),
      Actual = trimws(as.character(Actual)),
      Predicted = trimws(as.character(Predicted))
    )

  if (anyDuplicated(prediction_data$File_Name)) {
    stop("Duplicate File_Name values found in prediction file.")
  }

  # Verify that predictions correspond to the common test split.
  if (nrow(prediction_data) != nrow(test_data)) {
    stop(
      algorithm, " - ", task,
      ": prediction row count does not match the common test split. ",
      "Expected ", nrow(test_data), ", found ",
      nrow(prediction_data), "."
    )
  }

  if (!setequal(prediction_data$File_Name, test_data$File_Name)) {
    stop(
      algorithm, " - ", task,
      ": prediction filenames do not match the common test split."
    )
  }

  # Reorder predictions to match the common test split.
  prediction_data <- test_data %>%
    select(File_Name, Actual) %>%
    left_join(
      prediction_data %>% select(File_Name, Predicted),
      by = "File_Name"
    )

  if (anyNA(prediction_data$Predicted)) {
    stop(algorithm, " - ", task, ": missing predictions found.")
  }

  # Verify the saved Actual labels against the common split.
  # The common split is treated as the reference for evaluation.
  classes <- sort(unique(test_data$Actual))

  if (any(!prediction_data$Predicted %in% classes)) {
    stop(algorithm, " - ", task,
         ": predictions contain labels absent from the test classes.")
  }

  actual <- factor(prediction_data$Actual, levels = classes)
  predicted <- factor(prediction_data$Predicted, levels = classes)

  cm <- table(Actual = actual, Predicted = predicted)

  # Calculate class-wise precision, recall and F1.
  class_metrics <- lapply(classes, function(class_name) {

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
      Algorithm = algorithm,
      Task = task,
      Class = class_name,
      Support = sum(cm[class_name, ]),
      Precision = precision,
      Recall = recall,
      F1_Score = f1
    )
  }) %>%
    bind_rows()

  # Overall metrics.
  summary_row <- data.frame(
    Algorithm = algorithm,
    Task = task,
    Test_Records = sum(cm),
    Number_of_Classes = length(classes),
    Accuracy = sum(diag(cm)) / sum(cm),
    Balanced_Accuracy = mean(class_metrics$Recall),
    Macro_F1 = mean(class_metrics$F1_Score)
  )

  # Save confusion matrix for this algorithm and task.
  safe_algorithm <- str_replace_all(tolower(algorithm), "[^a-z0-9]+", "_")
  safe_task <- tolower(task)

  cm_output <- data.frame(
    Actual_Class = rownames(cm),
    as.data.frame.matrix(cm),
    check.names = FALSE
  )

  write_csv(
    cm_output,
    file.path(
      output_dir,
      paste0(safe_task, "_", safe_algorithm, "_confusion_matrix.csv")
    )
  )

  list(
    summary = summary_row,
    class_metrics = class_metrics
  )
}

# --------------------------- 6. Process all prediction files ----------

all_summaries <- list()
all_class_metrics <- list()
processed <- character()
skipped <- character()

for (file_path in prediction_files) {

  file_name <- basename(file_path)
  algorithm <- identify_algorithm(file_name)
  task <- identify_task(file_name)

  # Skip files that cannot be identified.
  if (is.na(algorithm) || is.na(task)) {
    skipped <- c(skipped, file_name)
    next
  }

  cat("\nReading:", file_name, "\n")
  cat("Algorithm:", algorithm, "| Task:", task, "\n")

  prediction_data <- read_csv(file_path, show_col_types = FALSE)

  reference_test <- if (task == "Species") {
    species_test
  } else {
    threat_test
  }

  result <- calculate_metrics(
    prediction_data = prediction_data,
    test_data = reference_test,
    algorithm = algorithm,
    task = task
  )

  all_summaries[[length(all_summaries) + 1]] <- result$summary
  all_class_metrics[[length(all_class_metrics) + 1]] <- result$class_metrics

  processed <- c(processed, paste(algorithm, task, sep = " - "))
}

# --------------------------- 7. Validate six algorithms ---------------

if (length(all_summaries) == 0) {
  stop(
    "No compatible prediction files were processed.\n",
    "Check prediction filenames and their columns."
  )
}

summary_table <- bind_rows(all_summaries)
class_metrics_table <- bind_rows(all_class_metrics)

expected_algorithms <- c(
  "SVM",
  "XGBoost",
  "Random Forest",
  "KNN",
  "Gaussian NB",
  "Decision Tree"
)

cat("\n========================================\n")
cat("PROCESSED MODEL/TASK COMBINATIONS:\n")
print(processed)

if (length(skipped) > 0) {
  cat("\nSkipped files (could not identify algorithm/task):\n")
  print(skipped)
}

# Check for missing algorithm/task combinations.
expected_combinations <- expand.grid(
  Algorithm = expected_algorithms,
  Task = c("Species", "Threat"),
  stringsAsFactors = FALSE
)

missing_combinations <- anti_join(
  expected_combinations,
  summary_table,
  by = c("Algorithm", "Task")
)

if (nrow(missing_combinations) > 0) {
  cat("\nWARNING: Some model results are missing:\n")
  print(missing_combinations)
}

# Check duplicate algorithm/task combinations.
duplicates <- summary_table %>%
  count(Algorithm, Task) %>%
  filter(n > 1)

if (nrow(duplicates) > 0) {
  stop(
    "Multiple prediction files were matched to the same algorithm/task. ",
    "Please remove old duplicate prediction files or adjust the file search."
  )
}

# --------------------------- 8. Save consolidated results --------------

summary_table <- summary_table %>%
  arrange(Task, desc(Macro_F1))

class_metrics_table <- class_metrics_table %>%
  arrange(Task, Algorithm, Class)

write_csv(
  summary_table,
  file.path(output_dir, "all_algorithms_comparison_summary.csv")
)

write_csv(
  class_metrics_table,
  file.path(output_dir, "all_algorithms_class_wise_metrics.csv")
)

# --------------------------- 9. Create comparison tables ---------------

macro_f1_table <- summary_table %>%
  select(Algorithm, Task, Macro_F1) %>%
  pivot_wider(names_from = Task, values_from = Macro_F1)

accuracy_table <- summary_table %>%
  select(Algorithm, Task, Accuracy) %>%
  pivot_wider(names_from = Task, values_from = Accuracy)

balanced_accuracy_table <- summary_table %>%
  select(Algorithm, Task, Balanced_Accuracy) %>%
  pivot_wider(names_from = Task, values_from = Balanced_Accuracy)

write_csv(
  macro_f1_table,
  file.path(output_dir, "macro_f1_comparison.csv")
)

write_csv(
  accuracy_table,
  file.path(output_dir, "accuracy_comparison.csv")
)

write_csv(
  balanced_accuracy_table,
  file.path(output_dir, "balanced_accuracy_comparison.csv")
)

# --------------------------- 10. Identify best models ------------------

best_models <- summary_table %>%
  group_by(Task) %>%
  slice_max(order_by = Macro_F1, n = 1, with_ties = TRUE) %>%
  ungroup()

write_csv(
  best_models,
  file.path(output_dir, "best_model_by_task.csv")
)

# --------------------------- 11. Print final results ------------------

cat("\n========================================\n")
cat("CONSOLIDATED MODEL COMPARISON\n")
cat("========================================\n")

print(
  summary_table %>%
    mutate(
      Accuracy = round(Accuracy * 100, 2),
      Balanced_Accuracy = round(Balanced_Accuracy * 100, 2),
      Macro_F1 = round(Macro_F1 * 100, 2)
    )
)

cat("\n========================================\n")
cat("MACRO-F1 COMPARISON (%)\n")
cat("========================================\n")

print(
  macro_f1_table %>%
    mutate(across(where(is.numeric), ~ round(.x * 100, 2)))
)

cat("\n========================================\n")
cat("BEST MODEL FOR EACH TASK (BASED ON MACRO-F1)\n")
cat("========================================\n")

print(
  best_models %>%
    mutate(
      Accuracy = round(Accuracy * 100, 2),
      Balanced_Accuracy = round(Balanced_Accuracy * 100, 2),
      Macro_F1 = round(Macro_F1 * 100, 2)
    )
)

cat("\n========================================\n")
cat("COMPARISON COMPLETE\n")
cat("========================================\n")
cat("Output folder:", output_dir, "\n")
cat("\nGenerated files:\n")
cat("- all_algorithms_comparison_summary.csv\n")
cat("- all_algorithms_class_wise_metrics.csv\n")
cat("- macro_f1_comparison.csv\n")
cat("- accuracy_comparison.csv\n")
cat("- balanced_accuracy_comparison.csv\n")
cat("- best_model_by_task.csv\n")
cat("- Individual confusion matrix CSV files\n")
