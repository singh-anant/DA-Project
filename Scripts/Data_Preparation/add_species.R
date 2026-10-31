# This script adds the species column to the final_ml_features.csv file, which contains the extracted features for each audio recording. The species information is obtained from the combined_land_mammals.csv file.
library(dplyr)
library(readr)

#Load the pre-extracted math features and the raw label data
features_df <- read_csv("Data_Processed/final_ml_features.csv", show_col_types = FALSE)
metadata_df <- read_csv(
  "Data_Processed/combined_land_mammals.csv",
  show_col_types = FALSE
)
#Recreate the File_Name format in the metadata so they match
metadata_df <- metadata_df %>%
  #creates a File_Name column using the recording ID.
  mutate(File_Name = paste0("XC", id, ".mp3")) %>%
  #keeps only File_Name and the species name, renaming en to Species.
  select(File_Name, Species = en)

# Join the Species column onto your features matrix
#left_join() keeps all rows from features_df. If a filename has no matching species in the metadata, 
#its Species value will be NA.
final_merged_df <- left_join(features_df, metadata_df, by = "File_Name")

#Rearrange the columns to put Species right next to Target_Label
#Moves the Species column immediately after File_Name. It does not change any values.
final_merged_df <- final_merged_df %>% relocate(Species, .after = File_Name)

# Save the updated CSV
write_csv(final_merged_df, "Data_Processed/final_ml_features_with_species.csv")

cat("✅ Success! Species added. Saved as Data_Processed/final_ml_features_with_species.csv\n")