# Load required packages
#dplyr is used for data manipulation.
#readr is used to read and write CSV files.
library(dplyr)
library(readr)

# Define the exact paths to your 5 raw CSV files
files <- c(
  "Data_Raw/Africa_Land_Mammals.csv",
  "Data_Raw/America_Land_Mammals.csv",
  "Data_Raw/Asia_Land_Mammals.csv",
  "Data_Raw/Australia_Land_Mammals.csv",
  "Data_Raw/Europe_Land_Mammals.csv"
)

# Read all files and combine them into one master table
# We read all columns as text (col_types = "c") to prevent data mismatch errors
#lapply() applies a function to each element of a list or vector.
combined_df <- lapply(files, function(f) {
  #Read every column as character/text
  read_csv(f, col_types = cols(.default = "c"),
  #Suppress the message that normally displays detected column types
   show_col_types = FALSE)
}) %>% 
  #bind_rows() stacks the data frames into one
  bind_rows()

#Remove any duplicate recordings based on the Xeno-canto 'id' column
final_df <- combined_df %>% 
  distinct(id, .keep_all = TRUE)

#Save the combined file into the processed data folder
write_csv(final_df, "Data_Processed/combined_land_mammals.csv")

# Print a confirmation message with the total row count
cat("Successfully combined", nrow(final_df), "unique recordings.\n")