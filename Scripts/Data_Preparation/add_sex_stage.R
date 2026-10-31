
# Import required libraries
library(readr)
library(dplyr)
library(stringr)

# Define file paths
feature_path <- "Data_Processed/final_ml_features_with_species.csv"
metadata_path <- "Data_Processed/combined_land_mammals.csv"
output_path <- "Data_Processed/final_ml_features_with_species.csv"

# Load datasets
features_df <- read_csv(
  feature_path,
  show_col_types = FALSE
)

metadata_df <- read_csv(
  metadata_path,
  col_types = cols(date = col_character()),
  show_col_types = FALSE
)

# Extract recording ID from File_Name (e.g., XC1158149.mp3 -> 1158149)
features_df <- features_df %>%
  mutate(
    Recording_ID = str_extract(File_Name, "(?<=XC)\\d+")
  )

# Prepare metadata lookup
metadata_lookup <- metadata_df %>%
  transmute(
    Recording_ID = as.character(id),
    sex = as.character(sex),
    stage = as.character(stage)
  ) %>%
  distinct()

# Stop if a recording ID has conflicting sex/stage metadata
conflicts <- metadata_lookup %>%
  count(Recording_ID) %>%
  filter(n > 1)

if (nrow(conflicts) > 0) {
  print(conflicts)
  stop("Some recording IDs have conflicting sex/stage metadata. Resolve these before joining.")
}

# Record original row count
original_rows <- nrow(features_df)

# Join sex and stage to the feature dataset
features_updated <- features_df %>%
  left_join(metadata_lookup, by = "Recording_ID")

# Verify that the join preserved every feature row
if (nrow(features_updated) != original_rows) {
  stop("Row count changed after joining metadata. Check recording IDs.")
}

# Remove helper column
features_updated <- features_updated %>%
  select(-Recording_ID)

# Overwrite the original file only after validation
write_csv(features_updated, output_path)

# Display summary
cat("Sex and stage columns added successfully.\n")
cat("Original feature rows:", original_rows, "\n")
cat("Updated feature rows:", nrow(features_updated), "\n")
cat("Missing sex values:", sum(is.na(features_updated$sex)), "\n")
cat("Missing stage values:", sum(is.na(features_updated$stage)), "\n")

print(features_updated %>% select(File_Name, Species, sex, stage) %>% head())
