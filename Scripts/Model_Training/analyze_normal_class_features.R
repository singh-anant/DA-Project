
# Analyze Acoustic Features Across Threat Classes

# Load required library
library(ggplot2)

# Read dataset
data <- read.csv(
  "Data_Processed/final_ml_features_richer_with_species_and_Threat.csv"
)

# Identify threat column
threat_candidates <- c("Threat_Level", "Threat", "Target_Label")
threat_col <- threat_candidates[threat_candidates %in% names(data)][1]

if (is.na(threat_col)) {
  stop("Threat column not found.")
}

# Select 33 audio features
basic_features <- c(
  "Pitch_Centroid",
  "Energy",
  "ZCR",
  "Spectral_Bandwidth",
  "Spectral_RollOff85",
  "Spectral_Flatness",
  "Spectral_Entropy"
)

mfcc_features <- grep("^MFCC", names(data), value = TRUE)

features <- c(basic_features, mfcc_features)

if (length(features) != 33) {
  stop("Expected 33 features. Found: ", length(features))
}

# Keep only known threat classes
class_names <- c("Dangerous", "Normal", "Weak")

data <- data[
  data[[threat_col]] %in% class_names,
]

# Remove missing and non-finite values
data <- data[
  complete.cases(data[, c(threat_col, features)]),
]

valid_rows <- apply(
  data[, features],
  1,
  function(x) all(is.finite(x))
)

data <- data[valid_rows, ]

data[[threat_col]] <- factor(
  data[[threat_col]],
  levels = class_names
)

# Create output folder
output_dir <- "Data_Processed/Normal_Feature_Analysis"

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

# ------------------------------------------------
# 1. Class distribution
# ------------------------------------------------

class_distribution <- as.data.frame(
  table(data[[threat_col]])
)

names(class_distribution) <- c("Threat_Class", "Count")

write.csv(
  class_distribution,
  paste0(output_dir, "/class_distribution.csv"),
  row.names = FALSE
)

# ------------------------------------------------
# 2. Calculate feature statistics
# ------------------------------------------------

feature_summary <- data.frame()

for (feature in features) {

  for (class_name in class_names) {

    values <- data[
      data[[threat_col]] == class_name,
      feature
    ]

    feature_summary <- rbind(
      feature_summary,
      data.frame(
        Feature = feature,
        Threat_Class = class_name,
        Count = length(values),
        Mean = mean(values),
        Median = median(values),
        SD = sd(values),
        IQR = IQR(values)
      )
    )
  }
}

write.csv(
  feature_summary,
  paste0(output_dir, "/feature_summary.csv"),
  row.names = FALSE
)

# ------------------------------------------------
# 3. Kruskal-Wallis test
# ------------------------------------------------

# Compare each feature across all three classes

statistical_results <- data.frame()

for (feature in features) {

  test_result <- kruskal.test(
    data[[feature]] ~ data[[threat_col]]
  )

  statistical_results <- rbind(
    statistical_results,
    data.frame(
      Feature = feature,
      Chi_Squared = as.numeric(test_result$statistic),
      P_Value = test_result$p.value
    )
  )
}

# Correct for multiple comparisons
statistical_results$Adjusted_P_Value <- p.adjust(
  statistical_results$P_Value,
  method = "BH"
)

# Sort by adjusted p-value
statistical_results <- statistical_results[
  order(statistical_results$Adjusted_P_Value),
]

write.csv(
  statistical_results,
  paste0(output_dir, "/kruskal_wallis_results.csv"),
  row.names = FALSE
)

# ------------------------------------------------
# 4. Compare Normal against other classes
# ------------------------------------------------

pairwise_results <- data.frame()

for (feature in features) {

  normal_values <- data[
    data[[threat_col]] == "Normal",
    feature
  ]

  dangerous_values <- data[
    data[[threat_col]] == "Dangerous",
    feature
  ]

  weak_values <- data[
    data[[threat_col]] == "Weak",
    feature
  ]

  # Normal vs Dangerous
  test1 <- wilcox.test(
    normal_values,
    dangerous_values,
    exact = FALSE
  )

  # Normal vs Weak
  test2 <- wilcox.test(
    normal_values,
    weak_values,
    exact = FALSE
  )

  pairwise_results <- rbind(
    pairwise_results,
    data.frame(
      Feature = feature,
      Comparison = "Normal vs Dangerous",
      P_Value = test1$p.value
    ),
    data.frame(
      Feature = feature,
      Comparison = "Normal vs Weak",
      P_Value = test2$p.value
    )
  )
}

# Correct for multiple comparisons
pairwise_results$Adjusted_P_Value <- p.adjust(
  pairwise_results$P_Value,
  method = "BH"
)

write.csv(
  pairwise_results,
  paste0(output_dir, "/normal_pairwise_tests.csv"),
  row.names = FALSE
)

# ------------------------------------------------
# 5. Prepare data for visualization
# ------------------------------------------------

long_data <- reshape(
  data,
  varying = features,
  v.names = "Value",
  timevar = "Feature",
  times = features,
  direction = "long"
)

names(long_data)[
  names(long_data) == threat_col
] <- "Threat_Class"

# Standardize each feature for visualization
long_data$Standardized_Value <- ave(
  long_data$Value,
  long_data$Feature,
  FUN = function(x) as.numeric(scale(x))
)

# ------------------------------------------------
# 6. Select top 8 features
# ------------------------------------------------

top_features <- head(
  statistical_results$Feature,
  8
)

top_data <- long_data[
  long_data$Feature %in% top_features,
]

# Create boxplots
feature_plot <- ggplot(
  top_data,
  aes(
    x = Threat_Class,
    y = Standardized_Value,
    fill = Threat_Class
  )
) +
  geom_boxplot(
    outlier.alpha = 0.3
  ) +
  facet_wrap(
    ~ Feature,
    scales = "free_y"
  ) +
  labs(
    title = "Acoustic Feature Comparison Across Threat Classes",
    subtitle = "Top 8 features selected by Kruskal-Wallis adjusted p-value",
    x = "Threat Class",
    y = "Standardized Feature Value"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 30, hjust = 1),
    legend.position = "none"
  )

ggsave(
  paste0(output_dir, "/top_8_feature_boxplots.png"),
  feature_plot,
  width = 12,
  height = 8
)

# ------------------------------------------------
# 7. Print important results
# ------------------------------------------------

cat("\n========== CLASS DISTRIBUTION ==========\n")
print(class_distribution)

cat("\n========== TOP 10 FEATURES ==========\n")
print(head(statistical_results, 10))

cat("\n========== NORMAL VS OTHER CLASSES ==========\n")
print(
  pairwise_results[
    order(pairwise_results$Adjusted_P_Value),
  ][1:10, ]
)

cat("\nAnalysis completed!\n")
cat("Results saved in:", output_dir, "\n")
