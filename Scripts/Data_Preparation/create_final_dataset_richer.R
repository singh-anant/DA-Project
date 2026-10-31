
# Create the final richer dataset with audio features, species, and threat labels.

library(readr)
library(dplyr)

# Input file containing the richer audio features
input_file <- "Data_Processed/final_ml_features_richer.csv"

# Existing file containing species and threat labels
metadata_file <- "Data_Processed/final_ml_features_with_threat.csv"

# New output file
output_file <- "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv"

# Load both datasets
features_df <- read_csv(input_file, show_col_types = FALSE)
metadata_df <- read_csv(metadata_file, show_col_types = FALSE)

# Required columns in the richer feature dataset
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

# Required metadata and label columns
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

# Check whether required columns exist
missing_features <- setdiff(feature_columns, names(features_df))
missing_metadata <- setdiff(metadata_columns, names(metadata_df))

if (length(missing_features) > 0) {
  stop(
    "Missing feature columns: ",
    paste(missing_features, collapse = ", ")
  )
}

if (length(missing_metadata) > 0) {
  stop(
    "Missing metadata columns: ",
    paste(missing_metadata, collapse = ", ")
  )
}

# Keep only the required columns
features_df <- features_df %>%
  select(all_of(feature_columns))

metadata_df <- metadata_df %>%
  select(all_of(metadata_columns))

# Check for duplicate filenames before merging
if (anyDuplicated(features_df$File_Name) > 0) {
  stop("Duplicate File_Name values found in richer feature dataset.")
}

if (anyDuplicated(metadata_df$File_Name) > 0) {
  stop("Duplicate File_Name values found in metadata dataset.")
}

# Merge richer audio features with species and threat information
final_df <- metadata_df %>%
  inner_join(features_df, by = "File_Name")

# Save the new dataset
write_csv(final_df, output_file)

# Display dataset information
cat("Richer final dataset created:", output_file, "\n")
cat("Total rows:", nrow(final_df), "\n")
cat("Total columns:", ncol(final_df), "\n\n")

# Display threat-level distribution
print(count(final_df, Threat_Level, sort = TRUE))
