
# Verify Threat Label Mapping

# Load required library
library(dplyr)
library(readr)

# Load final dataset
data <- read_csv(
  "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv",
  show_col_types = FALSE
)

# Create output folder
output_dir <- "Data_Processed/Threat_Analysis"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# 1. Count threat classes for each species
species_threat <- data %>%
  count(Species, Threat_Level) %>%
  arrange(Species, Threat_Level)

write_csv(
  species_threat,
  file.path(output_dir, "full_species_threat_mapping.csv")
)

# 2. Check whether each species has consistent DEFRA categories
species_defra <- data %>%
  group_by(Species) %>%
  summarise(
    Total_Records = n(),
    DEFRA_Categories = n_distinct(DEFRA_Category),
    .groups = "drop"
  ) %>%
  arrange(desc(DEFRA_Categories))

write_csv(
  species_defra,
  file.path(output_dir, "species_defra_consistency.csv")
)

# 3. Check DEFRA category and threat-level mapping
defra_threat <- data %>%
  count(DEFRA_Category, Threat_Level) %>%
  arrange(DEFRA_Category, Threat_Level)

write_csv(
  defra_threat,
  file.path(output_dir, "defra_threat_mapping.csv")
)

# Display results
cat("\n--- Threat Class Distribution ---\n")
print(data %>% count(Threat_Level))

cat("\n--- Species Appearing in Multiple Threat Classes ---\n")
print(
  species_threat %>%
    group_by(Species) %>%
    summarise(
      Number_of_Threat_Classes = n_distinct(Threat_Level),
      .groups = "drop"
    ) %>%
    filter(Number_of_Threat_Classes > 1)
)

cat("\n--- DEFRA Category vs Threat Level ---\n")
print(defra_threat)

cat("\nAll CSV files saved in:", output_dir, "\n")
