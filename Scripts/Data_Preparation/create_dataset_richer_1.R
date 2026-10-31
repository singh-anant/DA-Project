# =====================================================================
# Create Dataset: Richer Audio Features (Version 1)
#
# Combines:
#   1. Audio features from final_ml_features_richer_1.csv
#   2. Species and threat metadata from final_ml_features_with_threat.csv
#
# Output:
#   Data_Processed/final_ml_features_richer_1_with_species_and_Threat.csv
#
# The original datasets are not overwritten.
# =====================================================================

library(readr)
library(dplyr)

# --------------------------- 1. File paths -----------------------------

features_file <- "Data_Processed/final_ml_features_richer_1.csv"
metadata_file <- "Data_Processed/final_ml_features_with_threat.csv"

output_file <- paste0(
  "Data_Processed/",
  "final_ml_features_richer_1_with_species_and_Threat.csv"
)

# --------------------------- 2. Check input files ----------------------

if (!file.exists(features_file)) {
  stop("Richer audio feature file not found: ", features_file,
       "\nRun feature_extraction_richer_1.R first.")
}

if (!file.exists(metadata_file)) {
  stop("Metadata file not found: ", metadata_file,
       "\nCheck that final_ml_features_with_threat.csv exists.")
}

# --------------------------- 3. Read datasets --------------------------

features_df <- read_csv(features_file, show_col_types = FALSE)
metadata_df <- read_csv(metadata_file, show_col_types = FALSE)

# --------------------------- 4. Define required columns ----------------

# 3 original features + 4 additional spectral features
# + 13 MFCC means + 13 MFCC standard deviations = 33 features.
feature_columns <- c(
  "File_Name",
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

# Keep the existing species and threat metadata.
metadata_columns <- c(
  "File_Name",
  "Species",
  "sex",
  "stage",
  "Scientific_Name",
  "DEFRA_Category",
  "DEFRA_Category_Name",
  "Threat_Level"
)

# --------------------------- 5. Validate columns -----------------------

missing_features <- setdiff(feature_columns, names(features_df))
missing_metadata <- setdiff(metadata_columns, names(metadata_df))

if (length(missing_features) > 0) {
  stop(
    "The richer feature file is missing columns: ",
    paste(missing_features, collapse = ", ")
  )
}

if (length(missing_metadata) > 0) {
  stop(
    "The metadata file is missing columns: ",
    paste(missing_metadata, collapse = ", ")
  )
}

# --------------------------- 6. Select useful columns ------------------

features_df <- features_df %>%
  select(all_of(feature_columns)) %>%
  mutate(File_Name = as.character(File_Name))

metadata_df <- metadata_df %>%
  select(all_of(metadata_columns)) %>%
  mutate(File_Name = as.character(File_Name))

# Remove empty recording identifiers before merging.
features_df <- features_df %>%
  filter(!is.na(File_Name), File_Name != "")

metadata_df <- metadata_df %>%
  filter(!is.na(File_Name), File_Name != "")

# --------------------------- 7. Check duplicate recordings --------------

if (anyDuplicated(features_df$File_Name) > 0) {
  duplicate_names <- features_df %>%
    count(File_Name) %>%
    filter(n > 1) %>%
    pull(File_Name)

  stop(
    "Duplicate File_Name values found in richer features. Examples: ",
    paste(head(duplicate_names, 10), collapse = ", ")
  )
}

if (anyDuplicated(metadata_df$File_Name) > 0) {
  duplicate_names <- metadata_df %>%
    count(File_Name) %>%
    filter(n > 1) %>%
    pull(File_Name)

  stop(
    "Duplicate File_Name values found in metadata. Examples: ",
    paste(head(duplicate_names, 10), collapse = ", ")
  )
}

# --------------------------- 8. Merge datasets -------------------------

# Keep only recordings that have both richer audio features and metadata.
final_df <- metadata_df %>%
  inner_join(features_df, by = "File_Name")

if (nrow(final_df) == 0) {
  stop("The merge produced zero rows. Check that File_Name values match.")
}

# Arrange columns in a predictable order.
final_df <- final_df %>%
  select(
    File_Name,
    Species,
    sex,
    stage,
    Pitch_Centroid,
    Energy,
    ZCR,
    Spectral_Bandwidth,
    Spectral_RollOff85,
    Spectral_Flatness,
    Spectral_Entropy,
    all_of(paste0("MFCC_", 1:13, "_Mean")),
    all_of(paste0("MFCC_", 1:13, "_SD")),
    Scientific_Name,
    DEFRA_Category,
    DEFRA_Category_Name,
    Threat_Level
  )

# --------------------------- 9. Save new dataset -----------------------

write_csv(final_df, output_file)

# --------------------------- 10. Display dataset information ------------

matched_features <- sum(features_df$File_Name %in% metadata_df$File_Name)
features_without_metadata <- sum(!features_df$File_Name %in% metadata_df$File_Name)
metadata_without_features <- sum(!metadata_df$File_Name %in% features_df$File_Name)

cat("\n========================================\n")
cat("RICHER DATASET CREATION COMPLETE\n")
cat("Output file:", output_file, "\n")
cat("Rows:", nrow(final_df), "\n")
cat("Columns:", ncol(final_df), "\n")
cat("Numerical audio features:", length(feature_columns) - 1, "\n")
cat("Feature recordings without metadata:", features_without_metadata, "\n")
cat("Metadata recordings without richer features:", metadata_without_features, "\n\n")

cat("Threat-level distribution:\n")
print(count(final_df, Threat_Level, sort = TRUE))

cat("\nFirst few rows:\n")
print(head(final_df))
