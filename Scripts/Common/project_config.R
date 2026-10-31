# =====================================================================
# Project Configuration
# Central location for all project paths
# =====================================================================

# Project root directory
PROJECT_ROOT <- getwd()

# Main directories
DATA_DIR <- file.path(PROJECT_ROOT, "Data_Processed")

DATASET_DIR <- file.path(DATA_DIR, "Datasets")

SPLIT_DIR <- file.path(DATA_DIR, "Common_Splits")

MODEL_DIR <- file.path(DATA_DIR, "Models")

PREDICTION_DIR <- file.path(DATA_DIR, "Predictions")

EVALUATION_DIR <- file.path(DATA_DIR, "Evaluation")

FEATURE_IMPORTANCE_DIR <- file.path(
  DATA_DIR,
  "Feature_Importance"
)

FINAL_COMPARISON_DIR <- file.path(
  DATA_DIR,
  "Final_Model_Comparison"
)

# Check directories
required_dirs <- c(
  DATASET_DIR,
  SPLIT_DIR,
  MODEL_DIR,
  PREDICTION_DIR,
  EVALUATION_DIR,
  FEATURE_IMPORTANCE_DIR,
  FINAL_COMPARISON_DIR
)

for (folder in required_dirs) {

  if (!dir.exists(folder)) {
    stop("Directory not found: ", folder)
  }

}

cat("Project paths loaded successfully.\n")