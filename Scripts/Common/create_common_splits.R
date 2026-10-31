
# ============================================================
# STEP 1: CREATE COMMON TRAIN-TEST SPLITS
# Project: Threat Level Detection of Wild Animals
#
# Purpose:
# Create one fixed 80:20 stratified split for each task.
# All six algorithms will use these exact same records.
# ============================================================

# Install required packages if they are not already installed
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}

if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}

library(caret)
library(dplyr)

# ------------------------------------------------------------
# 1. Load the richer dataset
# ------------------------------------------------------------

data <- read.csv(
  "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv",
  stringsAsFactors = FALSE
)

cat("Original dataset:", nrow(data), "records\n")

# Check required columns
required_columns <- c("File_Name", "Species", "Threat_Level")

missing_columns <- setdiff(required_columns, colnames(data))

if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# Each audio recording must have a unique filename
if (anyDuplicated(data$File_Name)) {
  stop("Duplicate File_Name values found. Resolve them before splitting.")
}

# Create output directory
dir.create("Data_Processed/Common_Splits", showWarnings = FALSE)

# ------------------------------------------------------------
# 2. Function to create and save a stratified split
# ------------------------------------------------------------

create_task_split <- function(dataset, target_column, task_name, seed) {

  cat("\n========================================\n")
  cat("Creating split for:", task_name, "\n")
  cat("========================================\n")

  # Remove records with missing target or filename
  dataset <- dataset[
    !is.na(dataset[[target_column]]) &
      !is.na(dataset$File_Name) &
      dataset$File_Name != "",
  ]

  # Remove unknown labels
  dataset <- dataset[
    !tolower(trimws(dataset[[target_column]])) %in%
      c("unknown", "identity unknown", ""),
  ]

  # Convert target into a factor
  dataset[[target_column]] <- as.factor(dataset[[target_column]])

  # Additional filtering for Species:
  # Keep species having at least 20 recordings
  if (task_name == "Species") {

    species_counts <- table(dataset[[target_column]])

    eligible_species <- names(species_counts[species_counts >= 20])

    dataset <- dataset[
      dataset[[target_column]] %in% eligible_species,
    ]

    dataset[[target_column]] <- droplevels(
      dataset[[target_column]]
    )
  }

  # Check that enough records exist
  if (nrow(dataset) == 0) {
    stop("No records available for ", task_name)
  }

  cat("Total records:", nrow(dataset), "\n")
  cat("Number of classes:", nlevels(dataset[[target_column]]), "\n")

  # Display class distribution
  cat("\nClass distribution:\n")
  print(table(dataset[[target_column]]))

  # ----------------------------------------------------------
  # Stratified 80:20 train-test split
  # ----------------------------------------------------------

  set.seed(seed)

  train_indices <- createDataPartition(
    dataset[[target_column]],
    p = 0.80,
    list = FALSE
  )

  train_data <- dataset[train_indices, ]
  test_data <- dataset[-train_indices, ]

  # ----------------------------------------------------------
  # 3. Save training and testing datasets
  # ----------------------------------------------------------

  output_folder <- "Data_Processed/Common_Splits"

  write.csv(
    train_data,
    paste0(output_folder, "/", task_name, "_train.csv"),
    row.names = FALSE
  )

  write.csv(
    test_data,
    paste0(output_folder, "/", task_name, "_test.csv"),
    row.names = FALSE
  )

  # Save exact filenames separately
  write.csv(
    data.frame(File_Name = train_data$File_Name),
    paste0(output_folder, "/", task_name, "_train_filenames.csv"),
    row.names = FALSE
  )

  write.csv(
    data.frame(File_Name = test_data$File_Name),
    paste0(output_folder, "/", task_name, "_test_filenames.csv"),
    row.names = FALSE
  )

  # ----------------------------------------------------------
  # 4. Verify that train and test records do not overlap
  # ----------------------------------------------------------

  overlap <- intersect(
    train_data$File_Name,
    test_data$File_Name
  )

  if (length(overlap) > 0) {
    stop("Train-test overlap detected for ", task_name)
  }

  cat("\nTraining records:", nrow(train_data), "\n")
  cat("Testing records:", nrow(test_data), "\n")
  cat("Train-test overlap:", length(overlap), "\n")

  cat("\nSaved files for", task_name, "successfully.\n")

  return(invisible(list(
    train = train_data,
    test = test_data
  )))
}

# ------------------------------------------------------------
# 5. Create Species classification split
# ------------------------------------------------------------

species_split <- create_task_split(
  dataset = data,
  target_column = "Species",
  task_name = "Species",
  seed = 42
)

# ------------------------------------------------------------
# 6. Create Threat Level classification split
# ------------------------------------------------------------

threat_split <- create_task_split(
  dataset = data,
  target_column = "Threat_Level",
  task_name = "Threat",
  seed = 43
)

# ------------------------------------------------------------
# 7. Final summary
# ------------------------------------------------------------

cat("\n\n========================================\n")
cat("COMMON DATA SPLITS CREATED SUCCESSFULLY\n")
cat("========================================\n")

cat("\nSpecies:\n")
cat("Train:", nrow(species_split$train), "\n")
cat("Test :", nrow(species_split$test), "\n")

cat("\nThreat Level:\n")
cat("Train:", nrow(threat_split$train), "\n")
cat("Test :", nrow(threat_split$test), "\n")

cat("\nOutput folder: Data_Processed/Common_Splits/\n")
