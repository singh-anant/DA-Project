
# ============================================================
# THREAT DATA ANALYSIS
# Objective: Investigate poor Normal class classification
# ============================================================

# Import required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)

# ============================================================
# 1. LOAD TRAINING DATA
# ============================================================

data <- read.csv(
  "Data_Processed/Common_Splits/Threat_train.csv"
)

# Create output folder
output_dir <- "Data_Processed/Threat_Analysis"

dir.create(
  output_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

# Select acoustic features
feature_cols <- c(
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

features <- data[, feature_cols]

data$Threat_Level <- factor(
  data$Threat_Level,
  levels = c("Dangerous", "Normal", "Weak")
)

cat("Total training records:", nrow(data), "\n")


# ============================================================
# 2. CLASS DISTRIBUTION
# ============================================================

class_distribution <- as.data.frame(
  table(data$Threat_Level)
)

colnames(class_distribution) <- c(
  "Threat_Level", "Count"
)

print(class_distribution)

# Save class distribution
write.csv(
  class_distribution,
  file.path(output_dir, "class_distribution.csv"),
  row.names = FALSE
)

# Visualize class distribution
p1 <- ggplot(
  class_distribution,
  aes(x = Threat_Level, y = Count, fill = Threat_Level)
) +
  geom_col() +
  labs(
    title = "Threat Class Distribution",
    x = "Threat Class",
    y = "Number of Recordings"
  ) +
  theme_minimal()

ggsave(
  file.path(output_dir, "class_distribution.png"),
  p1,
  width = 7,
  height = 5
)


# ============================================================
# 3. SPECIES DISTRIBUTION IN EACH CLASS
# ============================================================

species_distribution <- data %>%
  group_by(Threat_Level, Species) %>%
  summarise(
    Recordings = n(),
    .groups = "drop"
  ) %>%
  arrange(Threat_Level, desc(Recordings))

print(species_distribution)

write.csv(
  species_distribution,
  file.path(output_dir, "species_distribution.csv"),
  row.names = FALSE
)

# Visualize species distribution
p2 <- ggplot(
  species_distribution,
  aes(x = Species, y = Recordings, fill = Threat_Level)
) +
  geom_col() +
  facet_wrap(~Threat_Level, scales = "free_x") +
  labs(
    title = "Species Distribution Across Threat Classes",
    x = "Species",
    y = "Number of Recordings"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(
      angle = 90,
      hjust = 1,
      size = 6
    )
  )

ggsave(
  file.path(output_dir, "species_distribution.png"),
  p2,
  width = 14,
  height = 8
)


# ============================================================
# 4. AVERAGE ACOUSTIC FEATURES PER CLASS
# ============================================================

mean_features <- data %>%
  group_by(Threat_Level) %>%
  summarise(
    across(
      all_of(feature_cols),
      ~ mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

print(mean_features)

write.csv(
  mean_features,
  file.path(output_dir, "mean_features.csv"),
  row.names = FALSE
)


# ============================================================
# 5. FEATURE DISTRIBUTION COMPARISON
# ============================================================

# Convert feature data into long format
feature_long <- data %>%
  select(Threat_Level, all_of(feature_cols)) %>%
  pivot_longer(
    cols = all_of(feature_cols),
    names_to = "Feature",
    values_to = "Value"
  )

# Create boxplots
p3 <- ggplot(
  feature_long,
  aes(x = Threat_Level, y = Value, fill = Threat_Level)
) +
  geom_boxplot(outlier.alpha = 0.4) +
  facet_wrap(~Feature, scales = "free", ncol = 4) +
  labs(
    title = "Acoustic Feature Distribution Across Threat Classes",
    x = "Threat Class",
    y = "Feature Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    strip.text = element_text(size = 7)
  )

ggsave(
  file.path(output_dir, "feature_boxplots.png"),
  p3,
  width = 16,
  height = 18
)


# ============================================================
# 6. NORMAL CLASS FEATURE STATISTICS
# ============================================================

normal_data <- data %>%
  filter(Threat_Level == "Normal")

normal_statistics <- data.frame(
  Feature = feature_cols,
  Mean = sapply(normal_data[, feature_cols], mean, na.rm = TRUE),
  SD = sapply(normal_data[, feature_cols], sd, na.rm = TRUE),
  Minimum = sapply(normal_data[, feature_cols], min, na.rm = TRUE),
  Maximum = sapply(normal_data[, feature_cols], max, na.rm = TRUE)
)

print(normal_statistics)

write.csv(
  normal_statistics,
  file.path(output_dir, "normal_class_statistics.csv"),
  row.names = FALSE
)


# ============================================================
# 7. CORRELATION BETWEEN ACOUSTIC FEATURES
# ============================================================

correlation_matrix <- cor(
  features,
  use = "pairwise.complete.obs"
)

png(
  file.path(output_dir, "feature_correlation.png"),
  width = 1200,
  height = 1000
)

corrplot(
  correlation_matrix,
  method = "color",
  type = "upper",
  tl.cex = 0.6,
  tl.col = "black"
)

dev.off()

# Save correlation matrix
write.csv(
  correlation_matrix,
  file.path(output_dir, "feature_correlation.csv")
)


# ============================================================
# 8. DETECT NORMAL CLASS OUTLIERS
# ============================================================

# Standardize using training data statistics
scaled_features <- scale(features)

normal_scaled <- scaled_features[
  data$Threat_Level == "Normal",
  ,
  drop = FALSE
]

# Calculate distance from Normal class mean
normal_center <- colMeans(
  normal_scaled,
  na.rm = TRUE
)

normal_distances <- sqrt(
  rowSums(
    sweep(
      normal_scaled,
      2,
      normal_center,
      "-"
    )^2
  )
)

# Identify records beyond mean + 2 SD distance
outlier_limit <- mean(normal_distances) +
  2 * sd(normal_distances)

outlier_indices <- which(
  normal_distances > outlier_limit
)

normal_outliers <- normal_data[
  outlier_indices,
  ,
  drop = FALSE
]

cat("\nNormal class outlier count:",
    length(outlier_indices), "\n")

print(normal_outliers)

write.csv(
  normal_outliers,
  file.path(output_dir, "normal_class_outliers.csv"),
  row.names = FALSE
)


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

cat("\n====================================\n")
cat("THREAT DATA ANALYSIS COMPLETED!\n")
cat("====================================\n")

cat("Total records:", nrow(data), "\n")
cat("Normal records:", nrow(normal_data), "\n")
cat("Number of acoustic features:", length(feature_cols), "\n")

cat("\nOutput saved in:", output_dir, "\n")
