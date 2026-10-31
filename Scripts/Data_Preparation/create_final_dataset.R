
# Create the final dataset with audio features, species, and threat labels.

library(readr)
library(dplyr)

input_file <- "Data_Processed/final_ml_features_with_threat.csv"
output_file <- "Data_Processed/final_ml_features_with_species_and_Threat.csv"

df <- read_csv(input_file, show_col_types = FALSE)

required_columns <- c(
  "File_Name", "Species", "sex", "stage",
  "Pitch_Centroid", "Energy", "ZCR",
  paste0("MFCC_", 1:13),
  "Scientific_Name", "DEFRA_Category",
  "DEFRA_Category_Name", "Threat_Level"
)

missing_columns <- setdiff(required_columns, names(df))

if (length(missing_columns) > 0) {
  stop("Missing columns: ", paste(missing_columns, collapse = ", "))
}

# Keep the useful project columns and preserve all recordings.
final_df <- df %>%
  select(all_of(required_columns))

write_csv(final_df, output_file)

cat("Final dataset created:", output_file, "\n")
cat("Rows:", nrow(final_df), "\n")
cat("Columns:", ncol(final_df), "\n\n")

print(count(final_df, Threat_Level, sort = TRUE))
