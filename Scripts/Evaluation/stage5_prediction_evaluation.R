# =====================================================================
# Stage 5: Final Prediction, DEFRA Mapping, and Evaluation
#
# Uses:
#   1. Species SVM model
#   2. Threat SVM model
#   3. 33 richer audio features
#
# Tasks:
#   - Evaluate saved test predictions
#   - Predict species from a new audio recording
#   - Map predicted species to DEFRA category
#   - Predict threat level directly using the Threat SVM
#
# Run from project root:
# Rscript Scripts/stage5_prediction_evaluation.R
#
# For a new recording:
# Rscript Scripts/stage5_prediction_evaluation.R "/path/audio.mp3"
# ======================================================================


# --------------------------- 1. Load libraries -------------------------

required_packages <- c(
  "readr",
  "dplyr",
  "e1071",
  "tuneR",
  "seewave",
  "av"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  stop(
    "Install these packages first: ",
    paste(missing_packages, collapse = ", ")
  )
}

library(readr)
library(dplyr)
library(e1071)
library(tuneR)
library(seewave)
library(av)


# --------------------------- 2. Paths and settings ---------------------

data_dir <- "Data_Processed"

# Final selected SVM models.

species_model_file <- file.path(
  data_dir,
  "species_richer_svm_svm_rbf.rds"
)

threat_model_file <- file.path(
  data_dir,
  "threat_richer_svm_svm_rbf.rds"
)

# Species to DEFRA mapping file.

mapping_file <- file.path(
  data_dir,
  "species_threat_mapping.csv"
)

# Saved held-out test predictions from Stage 6 .

species_test_file <- file.path(
  data_dir,
  "species_richer_svm_svm_test_predictions.csv"
)

threat_test_file <- file.path(
  data_dir,
  "threat_richer_svm_svm_test_predictions.csv"
)

# Stage 5 output files.

evaluation_file <- file.path(
  data_dir,
  "stage5_svm_evaluation_summary.csv"
)

class_metrics_file <- file.path(
  data_dir,
  "stage5_svm_class_wise_metrics.csv"
)

species_cm_file <- file.path(
  data_dir,
  "stage5_species_svm_confusion_matrix.csv"
)

threat_cm_file <- file.path(
  data_dir,
  "stage5_threat_svm_confusion_matrix.csv"
)

prediction_file <- file.path(
  data_dir,
  "stage5_svm_prediction.csv"
)


# --------------------------- 3. Define 33 features ---------------------

features <- c(
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

# Audio preprocessing settings.

TARGET_SAMPLE_RATE <- 22050
FRAME_LENGTH <- 1024
HOP_LENGTH <- 512
ROLLOFF_PERCENT <- 0.85

if (length(features) != 33) {
  stop("Expected exactly 33 audio features.")
}

dir.create(
  data_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# --------------------------- 4. Load models and mapping ----------------

required_files <- c(
  species_model_file,
  threat_model_file,
  mapping_file
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0) {
  stop(
    "Required files are missing:\n",
    paste(missing_files, collapse = "\n")
  )
}

# Load trained models.

species_model <- readRDS(species_model_file)

threat_model <- readRDS(threat_model_file)

# Load DEFRA mapping.

mapping <- read_csv(
  mapping_file,
  show_col_types = FALSE
)

# Check model types.

if (!inherits(species_model, "svm")) {
  stop("Species model is not an SVM model.")
}

if (!inherits(threat_model, "svm")) {
  stop("Threat model is not an SVM model.")
}

# Required mapping columns.

required_mapping <- c(
  "Species",
  "DEFRA_Category",
  "DEFRA_Category_Name",
  "Threat_Level",
  "Match_Level",
  "Audit_Status"
)

missing_columns <- setdiff(
  required_mapping,
  names(mapping)
)

if (length(missing_columns) > 0) {
  stop(
    "Mapping file is missing columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# Normalize species names for matching.

normalise_name <- function(x) {

  tolower(
    gsub("\\s+", " ", trimws(as.character(x)))
  )

}

mapping <- mapping %>%
  mutate(
    Species_Key = normalise_name(Species)
  )

# Check duplicate species mappings.

if (anyDuplicated(mapping$Species_Key) > 0) {
  stop(
    "Duplicate species names found in DEFRA mapping."
  )
}

cat("\nModels and DEFRA mapping loaded successfully.\n")
# --------------------------- 5. Calculate spectral features ------------

calculate_spectral_features <- function(
    samples,
    sample_rate,
    frame_length = FRAME_LENGTH,
    hop_length = HOP_LENGTH,
    rolloff_percent = ROLLOFF_PERCENT) {

  # Convert audio samples into numeric values.

  x <- as.numeric(samples)

  x <- x[is.finite(x)]

  if (length(x) == 0) {
    stop("No valid audio samples found.")
  }

  # Remove DC offset.

  x <- x - mean(x)

  # Pad very short audio recordings.

  if (length(x) < frame_length) {

    x <- c(
      x,
      rep(0, frame_length - length(x))
    )

  }

  # Calculate frame starting positions.

  frame_starts <- seq(
    1,
    length(x) - frame_length + 1,
    by = hop_length
  )

  # Apply Hann window.

  hann_window <- 0.5 - 0.5 * cos(
    2 * pi * (0:(frame_length - 1)) /
      (frame_length - 1)
  )

  number_of_bins <- frame_length %/% 2 + 1

  power_sum <- numeric(number_of_bins)

  # Calculate FFT for every frame.

  for (frame_start in frame_starts) {

    frame_indices <- frame_start:
      (frame_start + frame_length - 1)

    frame <- x[frame_indices] * hann_window

    fft_values <- fft(frame)

    power_sum <- power_sum +
      Mod(fft_values[seq_len(number_of_bins)])^2

  }

  # Average spectral power.

  mean_power <- power_sum / length(frame_starts)

  frequencies <- (0:(number_of_bins - 1)) *
    sample_rate / frame_length

  total_power <- sum(mean_power)

  if (!is.finite(total_power) || total_power <= 0) {
    stop("Audio is silent or has no measurable spectral power.")
  }

  # Calculate spectral magnitude.

  magnitude <- sqrt(mean_power)

  total_magnitude <- sum(magnitude)

  # Spectral centroid.

  spectral_centroid <- sum(
    frequencies * magnitude
  ) / total_magnitude

  # Spectral bandwidth.

  spectral_bandwidth <- sqrt(
    sum(
      magnitude *
        (frequencies - spectral_centroid)^2
    ) / total_magnitude
  )

  # Spectral roll-off at 85%.

  cumulative_power <- cumsum(mean_power) / total_power

  rolloff_index <- which(
    cumulative_power >= rolloff_percent
  )[1]

  spectral_rolloff <- frequencies[rolloff_index]

  # Spectral flatness.

  epsilon <- max(mean_power) * 1e-12 +
    .Machine$double.eps

  power_for_log <- mean_power + epsilon

  spectral_flatness <- exp(
    mean(log(power_for_log))
  ) / mean(power_for_log)

  # Spectral entropy.

  probability <- mean_power / total_power

  positive_probability <- probability[
    probability > 0
  ]

  spectral_entropy <- -sum(
    positive_probability *
      log2(positive_probability)
  ) / log2(length(probability))

  # Return calculated features.

  return(
    c(
      Spectral_Centroid = spectral_centroid,
      Spectral_Bandwidth = spectral_bandwidth,
      Spectral_RollOff85 = spectral_rolloff,
      Spectral_Flatness = spectral_flatness,
      Spectral_Entropy = spectral_entropy
    )
  )

}
# --------------------------- 6. Extract all 33 features -----------------

extract_features <- function(audio_path) {

  # Check audio file.

  if (!file.exists(audio_path)) {

    stop(
      "Audio file not found: ",
      audio_path
    )

  }

  # Temporary WAV file.

  temporary_wav <- tempfile(
    fileext = ".wav"
  )

  # Automatically delete temporary file after processing.

  on.exit({

    if (file.exists(temporary_wav)) {
      unlink(temporary_wav)
    }

  }, add = TRUE)

  # Convert audio to mono and 22050 Hz.

  av_audio_convert(
    audio = audio_path,
    output = temporary_wav,
    channels = 1,
    sample_rate = TARGET_SAMPLE_RATE,
    verbose = FALSE
  )

  # Read converted audio.

  wave <- tuneR::readWave(
    temporary_wav
  )

  samples <- as.numeric(wave@left)

  sample_rate <- wave@samp.rate

  if (length(samples) == 0) {
    stop("Audio contains no readable samples.")
  }

  if (sample_rate != TARGET_SAMPLE_RATE) {
    stop("Incorrect sample rate after conversion.")
  }

  # ----------------------------------------------------------
  # 6.1 RMS Energy
  # ----------------------------------------------------------

  energy <- sqrt(
    mean(samples^2, na.rm = TRUE)
  )

  # ----------------------------------------------------------
  # 6.2 Zero Crossing Rate
  # ----------------------------------------------------------

  zcr_values <- seewave::zcr(
    wave,
    f = sample_rate,
    wl = FRAME_LENGTH,
    plot = FALSE
  )

  mean_zcr <- mean(
    zcr_values[, 2],
    na.rm = TRUE
  )

  # ----------------------------------------------------------
  # 6.3 Spectral Features
  # ----------------------------------------------------------

  spectral <- calculate_spectral_features(
    samples,
    sample_rate
  )

  # ----------------------------------------------------------
  # 6.4 MFCC Features
  # ----------------------------------------------------------

  mfcc_matrix <- tuneR::melfcc(
    wave,
    sr = sample_rate,
    numcep = 13,
    wintime = 0.025,
    hoptime = 0.010,
    frames_in_rows = TRUE
  )

  mfcc_matrix <- as.matrix(mfcc_matrix)

  if (ncol(mfcc_matrix) != 13) {
    stop("Expected 13 MFCC coefficients.")
  }

  # Mean of each MFCC coefficient.

  mfcc_means <- colMeans(
    mfcc_matrix,
    na.rm = TRUE
  )

  # Standard deviation of each MFCC coefficient.

  mfcc_sds <- apply(
    mfcc_matrix,
    2,
    sd,
    na.rm = TRUE
  )

  mfcc_sds[!is.finite(mfcc_sds)] <- 0

  # ----------------------------------------------------------
  # 6.5 Create final feature dataframe
  # ----------------------------------------------------------

  result <- data.frame(

    Pitch_Centroid = as.numeric(
      spectral["Spectral_Centroid"]
    ),

    Energy = as.numeric(energy),

    ZCR = as.numeric(mean_zcr),

    Spectral_Bandwidth = as.numeric(
      spectral["Spectral_Bandwidth"]
    ),

    Spectral_RollOff85 = as.numeric(
      spectral["Spectral_RollOff85"]
    ),

    Spectral_Flatness = as.numeric(
      spectral["Spectral_Flatness"]
    ),

    Spectral_Entropy = as.numeric(
      spectral["Spectral_Entropy"]
    )

  )

  # Add 13 MFCC means.

  for (i in 1:13) {

    result[[paste0("MFCC_", i, "_Mean")]] <-
      as.numeric(mfcc_means[i])

  }

  # Add 13 MFCC standard deviations.

  for (i in 1:13) {

    result[[paste0("MFCC_", i, "_SD")]] <-
      as.numeric(mfcc_sds[i])

  }

  # Maintain exact feature order used during training.

  result <- result[
    ,
    features,
    drop = FALSE
  ]

  # Validate extracted features.

  if (
    ncol(result) != 33 ||
    any(!is.finite(as.matrix(result)))
  ) {

    stop(
      "Feature extraction returned invalid values."
    )

  }

  cat("\nSuccessfully extracted 33 audio features.\n")

  return(result)

}
# --------------------------- 7. Evaluate saved predictions --------------

evaluate_saved_predictions <- function(
    prediction_path,
    task_name) {

  # Check prediction file.

  if (!file.exists(prediction_path)) {

    message(
      "Prediction file not found: ",
      prediction_path
    )

    return(NULL)

  }

  # Load predictions.

  prediction_data <- read_csv(
    prediction_path,
    show_col_types = FALSE
  )

  # Validate columns.

  required_columns <- c(
    "Actual",
    "Predicted"
  )

  if (!all(required_columns %in% names(prediction_data))) {

    stop(
      "Actual and Predicted columns are missing in ",
      prediction_path
    )

  }

  prediction_data <- prediction_data %>%
    filter(
      !is.na(Actual),
      !is.na(Predicted)
    )

  actual <- trimws(
    as.character(prediction_data$Actual)
  )

  predicted <- trimws(
    as.character(prediction_data$Predicted)
  )

  # Get all class names.

  class_names <- sort(
    unique(c(actual, predicted))
  )

  # Create confusion matrix.

  cm <- table(
    Actual = factor(
      actual,
      levels = class_names
    ),

    Predicted = factor(
      predicted,
      levels = class_names
    )
  )

  # ----------------------------------------------------------
  # Calculate class-wise metrics
  # ----------------------------------------------------------

  class_metrics <- lapply(
    class_names,
    function(class_name) {

      TP <- cm[class_name, class_name]

      FP <- sum(cm[, class_name]) - TP

      FN <- sum(cm[class_name, ]) - TP

      support <- sum(cm[class_name, ])

      # Precision.

      precision <- if ((TP + FP) == 0) {
        0
      } else {
        as.numeric(TP / (TP + FP))
      }

      # Recall.

      recall <- if ((TP + FN) == 0) {
        0
      } else {
        as.numeric(TP / (TP + FN))
      }

      # F1-score.

      f1 <- if ((2 * TP + FP + FN) == 0) {
        0
      } else {
        as.numeric(
          2 * TP / (2 * TP + FP + FN)
        )
      }

      data.frame(

        Task = task_name,

        Class = class_name,

        Support = support,

        Precision = precision,

        Recall = recall,

        F1_Score = f1

      )

    }
  ) %>%
    bind_rows()

  # ----------------------------------------------------------
  # Calculate overall evaluation metrics
  # ----------------------------------------------------------

  accuracy <- mean(
    actual == predicted
  )

  balanced_accuracy <- mean(
    class_metrics$Recall[
      class_metrics$Support > 0
    ]
  )

  macro_f1 <- mean(
    class_metrics$F1_Score[
      class_metrics$Support > 0
    ]
  )

  summary <- data.frame(

    Task = task_name,

    Test_Records = length(actual),

    Classes = length(unique(actual)),

    Accuracy = accuracy,

    Balanced_Accuracy = balanced_accuracy,

    Macro_F1 = macro_f1

  )

  # ----------------------------------------------------------
  # Save confusion matrix
  # ----------------------------------------------------------

  confusion_output <- data.frame(

    Actual_Class = rownames(cm),

    as.data.frame.matrix(cm),

    check.names = FALSE

  )

  if (task_name == "Species") {

    write_csv(
      confusion_output,
      species_cm_file
    )

  } else {

    write_csv(
      confusion_output,
      threat_cm_file
    )

  }

  return(
    list(
      summary = summary,
      class_metrics = class_metrics
    )
  )

}


# --------------------------- 8. Run evaluation --------------------------

cat("\n========================================\n")

cat("STAGE 5: SVM MODEL EVALUATION\n")

cat("========================================\n")


# Evaluate Species SVM.

species_evaluation <- evaluate_saved_predictions(
  species_test_file,
  "Species"
)


# Evaluate Threat SVM.

threat_evaluation <- evaluate_saved_predictions(
  threat_test_file,
  "Threat"
)


# Collect evaluation results.

evaluation_rows <- list()

class_metric_rows <- list()


if (!is.null(species_evaluation)) {

  evaluation_rows[[1]] <- species_evaluation$summary

  class_metric_rows[[1]] <- species_evaluation$class_metrics

}


if (!is.null(threat_evaluation)) {

  evaluation_rows[[length(evaluation_rows) + 1]] <-
    threat_evaluation$summary

  class_metric_rows[[length(class_metric_rows) + 1]] <-
    threat_evaluation$class_metrics

}


# Save final evaluation files.

if (length(evaluation_rows) > 0) {

  evaluation_summary <- bind_rows(
    evaluation_rows
  )

  class_metrics_summary <- bind_rows(
    class_metric_rows
  )

  write_csv(
    evaluation_summary,
    evaluation_file
  )

  write_csv(
    class_metrics_summary,
    class_metrics_file
  )

  cat("\nEvaluation Summary:\n")

  print(evaluation_summary)

  cat("\nSaved evaluation summary:\n")
  cat(evaluation_file, "\n")

  cat("\nSaved class-wise metrics:\n")
  cat(class_metrics_file, "\n")

  cat("\nSaved Species confusion matrix:\n")
  cat(species_cm_file, "\n")

  cat("\nSaved Threat confusion matrix:\n")
  cat(threat_cm_file, "\n")

}
# --------------------------- 9. Predict new audio recording -------------

args <- commandArgs(
  trailingOnly = TRUE
)


# Check whether an audio file was provided.

if (
  length(args) == 0 ||
  trimws(args[1]) == ""
) {

  cat(
    "\nEvaluation completed successfully.\n\n",

    "To predict a new recording, run:\n",

    'Rscript Scripts/stage5_prediction_evaluation.R "/full/path/to/audio.mp3"\n',

    sep = ""
  )

} else {

  # Read audio path.

  audio_path <- args[1]

  cat("\n========================================\n")

  cat("STAGE 5: NEW AUDIO PREDICTION\n")

  cat("========================================\n")


  # ----------------------------------------------------------
  # 9.1 Extract 33 audio features
  # ----------------------------------------------------------

  cat("\nProcessing audio:", audio_path, "\n")

  new_features <- extract_features(
    audio_path
  )


  # ----------------------------------------------------------
  # 9.2 Predict Species
  # ----------------------------------------------------------

  species_prediction <- predict(
    species_model,
    newdata = new_features
  )

  predicted_species <- as.character(
    species_prediction
  )[1]


  # ----------------------------------------------------------
  # 9.3 DEFRA species mapping
  # ----------------------------------------------------------

  species_key <- normalise_name(
    predicted_species
  )

  matched_mapping <- mapping %>%
    filter(
      Species_Key == species_key
    )


  # Default values when mapping is unavailable.

  defra_category <- NA_character_

  defra_category_name <- "Unknown / unresolved"

  mapped_threat_label <- "Unknown"

  mapping_match_level <- "No unique species mapping found"

  mapping_audit_status <- "Needs review: no unique mapping"


  # Check whether exactly one mapping exists.

  if (nrow(matched_mapping) == 1) {

    candidate_category <- as.character(
      matched_mapping$DEFRA_Category[1]
    )

    candidate_category_name <- as.character(
      matched_mapping$DEFRA_Category_Name[1]
    )

    candidate_threat_label <- as.character(
      matched_mapping$Threat_Level[1]
    )


    # Check mapping validity.

    category_is_valid <-
      !is.na(candidate_category) &&

      nzchar(trimws(candidate_category)) &&

      !is.na(candidate_category_name) &&

      nzchar(trimws(candidate_category_name)) &&

      candidate_category_name != "Unknown / unresolved"


    if (category_is_valid) {

      defra_category <- candidate_category

      defra_category_name <- candidate_category_name

      if (
        !is.na(candidate_threat_label) &&
        nzchar(candidate_threat_label)
      ) {

        mapped_threat_label <- candidate_threat_label

      }

      mapping_match_level <- as.character(
        matched_mapping$Match_Level[1]
      )

      mapping_audit_status <- as.character(
        matched_mapping$Audit_Status[1]
      )

    } else {

      mapping_match_level <- as.character(
        matched_mapping$Match_Level[1]
      )

      mapping_audit_status <- as.character(
        matched_mapping$Audit_Status[1]
      )

    }

  }


  # ----------------------------------------------------------
  # 9.4 Direct Threat SVM prediction
  # ----------------------------------------------------------

  threat_prediction <- predict(
    threat_model,
    newdata = new_features
  )

  direct_threat_prediction <- as.character(
    threat_prediction
  )[1]


  # ----------------------------------------------------------
  # 9.5 Create final prediction output
  # ----------------------------------------------------------

  output <- data.frame(

    Audio_File = basename(audio_path),

    Predicted_Species = predicted_species,

    DEFRA_Category = defra_category,

    DEFRA_Category_Name = defra_category_name,

    Species_Mapped_Threat_Label = mapped_threat_label,

    Mapping_Match_Level = mapping_match_level,

    Mapping_Audit_Status = mapping_audit_status,

    Direct_Threat_Model_Prediction = direct_threat_prediction,

    Prediction_Model = "RBF SVM",

    stringsAsFactors = FALSE

  )


  # Save prediction.

  write_csv(
    output,
    prediction_file,
    na = ""
  )


  # ----------------------------------------------------------
  # 9.6 Display prediction results
  # ----------------------------------------------------------

  cat("\n========================================\n")

  cat("FINAL PREDICTION RESULTS\n")

  cat("========================================\n")


  cat(
    "\nAudio file:",
    basename(audio_path),
    "\n"
  )

  cat(
    "\nPredicted Species:",
    predicted_species,
    "\n"
  )

  cat(
    "\nDEFRA Category:",
    defra_category_name,
    "\n"
  )

  cat(
    "\nSpecies Mapped Threat Label:",
    mapped_threat_label,
    "\n"
  )

  cat(
    "\nDirect Threat SVM Prediction:",
    direct_threat_prediction,
    "\n"
  )


  cat("\nSaved prediction to:\n")

  cat(prediction_file, "\n")


  cat(
    "\nNote: Species-mapped threat label and direct Threat SVM prediction are separate outputs.\n"
  )

  cat(
    "DEFRA categories concern hazards to visitors in zoological gardens; they are not official acoustic danger labels for wild animals.\n"
  )

}

cat("\nStage 5 completed.\n")
