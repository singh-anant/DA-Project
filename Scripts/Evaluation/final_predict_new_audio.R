# ============================================================
# FINAL WILDLIFE AUDIO PREDICTION
# Predicts Species and Threat Level for one new audio recording
# using the saved combined SVM model bundle.
#
# Run from the DA-Project root directory:
#   Rscript Scripts/final_predict_new_audio.R "/path/to/audio.mp3"
#
# Or run the script in RStudio and select the audio file when
# prompted.
#
# Required model:
#   Models/final_wildlife_classification.rds
#
# This script does not create CSV files or modify the saved model.
# ============================================================

# ------------------------------------------------------------
# 1. LOAD REQUIRED PACKAGES
# ------------------------------------------------------------

required_packages <- c("e1071", "tuneR", "av")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install the following package(s) before running this script: ",
    paste(missing_packages, collapse = ", "),
    "\nExample: install.packages(c(",
    paste(sprintf("\"%s\"", missing_packages), collapse = ", "),
    "))"
  )
}

suppressPackageStartupMessages({
  library(e1071)
  library(tuneR)
  library(av)
})

# ------------------------------------------------------------
# 2. CONFIGURATION
# ------------------------------------------------------------

MODEL_FILE <- file.path(
  "Models",
  "final_wildlife_classification.rds"
)

TARGET_SAMPLE_RATE <- 22050
FRAME_LENGTH <- 2048
MFCC_COUNT <- 13

FEATURE_NAMES <- c(
  "Pitch_Centroid",
  "Energy",
  "ZCR",
  "Spectral_Bandwidth",
  "Spectral_RollOff85",
  "Spectral_Flatness",
  "Spectral_Entropy",
  paste0("MFCC_", 1:MFCC_COUNT, "_Mean"),
  paste0("MFCC_", 1:MFCC_COUNT, "_SD")
)

# ------------------------------------------------------------
# 3. SELECT THE NEW AUDIO FILE
# ------------------------------------------------------------

arguments <- commandArgs(trailingOnly = TRUE)

if (length(arguments) >= 1 && nzchar(arguments[1])) {
  audio_path <- arguments[1]
} else if (interactive()) {
  audio_path <- file.choose()
} else {
  stop(
    "Provide the audio file path when running this script.\n",
    "Example:\n",
    "Rscript Scripts/final_predict_new_audio.R \"/path/to/audio.mp3\""
  )
}

if (!file.exists(audio_path)) {
  stop("Audio file not found: ", audio_path)
}

audio_path <- normalizePath(audio_path, winslash = "/", mustWork = TRUE)

# ------------------------------------------------------------
# 4. LOAD THE SAVED MODEL BUNDLE
# ------------------------------------------------------------

if (!file.exists(MODEL_FILE)) {
  stop(
    "Saved model not found: ", MODEL_FILE,
    "\nRun final_wildlife_classification.R first to create the RDS model."
  )
}

model_bundle <- readRDS(MODEL_FILE)

# Use the two model objects stored in the RDS directly.
# This avoids dependency on the prediction closure saved in an older RDS.
if (is.null(model_bundle$species$model) ||
    is.null(model_bundle$threat$model)) {
  stop(
    "The RDS file must contain both species$model and threat$model. ",
    "Please check that Models/final_wildlife_classification.rds " ,
    "is the combined model bundle."
  )
}

if (is.null(model_bundle$feature_names) ||
    !identical(model_bundle$feature_names, FEATURE_NAMES)) {
  stop(
    "The feature list in the RDS model does not match this script. ",
    "Use the prediction script created for the same model version."
  )
}

# ------------------------------------------------------------
# 5. CALCULATE SPECTRAL FEATURES
#    These calculations match feature_extraction_richer_1.R.
# ------------------------------------------------------------

calculate_spectral_features <- function(signal, sample_rate) {

  signal <- signal - mean(signal)

  window <- 0.5 - 0.5 * cos(
    2 * pi * (0:(length(signal) - 1)) / (length(signal) - 1)
  )

  windowed_signal <- signal * window
  fft_values <- fft(windowed_signal)

  positive_count <- floor(length(fft_values) / 2) + 1
  magnitude <- Mod(fft_values[1:positive_count])
  power <- magnitude^2

  frequencies <- (0:(positive_count - 1)) *
    sample_rate / length(signal)

  # Ignore the zero-frequency component.
  power <- power[-1]
  frequencies <- frequencies[-1]

  total_power <- sum(power)

  if (!is.finite(total_power) || total_power <= 0) {
    stop("Audio is silent or has no measurable spectral power.")
  }

  probabilities <- power / total_power

  centroid <- sum(frequencies * probabilities)

  bandwidth <- sqrt(
    sum(probabilities * (frequencies - centroid)^2)
  )

  cumulative_power <- cumsum(probabilities)
  rolloff_index <- which(cumulative_power >= 0.85)[1]
  rolloff_85 <- frequencies[rolloff_index]

  epsilon <- .Machine$double.eps
  flatness <- exp(mean(log(power + epsilon))) /
    mean(power + epsilon)

  entropy <- -sum(
    probabilities * log(probabilities + epsilon)
  ) / log(length(probabilities))

  c(
    Pitch_Centroid = centroid,
    Spectral_Bandwidth = bandwidth,
    Spectral_RollOff85 = rolloff_85,
    Spectral_Flatness = flatness,
    Spectral_Entropy = entropy
  )
}

# ------------------------------------------------------------
# 6. CALCULATE MFCC MEAN AND STANDARD DEVIATION
#    This matches feature_extraction_richer_1.R.
# ------------------------------------------------------------

calculate_mfcc_features <- function(wave_object) {

  mfcc_values <- tuneR::melfcc(
    wave_object,
    numcep = MFCC_COUNT
  )

  mfcc_matrix <- as.matrix(mfcc_values)

  if (nrow(mfcc_matrix) != MFCC_COUNT &&
      ncol(mfcc_matrix) == MFCC_COUNT) {
    mfcc_matrix <- t(mfcc_matrix)
  }

  if (nrow(mfcc_matrix) != MFCC_COUNT) {
    stop("Unexpected MFCC dimensions returned by melfcc().")
  }

  mfcc_means <- rowMeans(mfcc_matrix, na.rm = TRUE)
  mfcc_sds <- apply(mfcc_matrix, 1, sd, na.rm = TRUE)
  mfcc_sds[!is.finite(mfcc_sds)] <- 0

  result <- c(mfcc_means, mfcc_sds)

  names(result) <- c(
    paste0("MFCC_", 1:MFCC_COUNT, "_Mean"),
    paste0("MFCC_", 1:MFCC_COUNT, "_SD")
  )

  result
}

# ------------------------------------------------------------
# 7. EXTRACT ALL 33 FEATURES FROM THE NEW AUDIO
# ------------------------------------------------------------

extract_audio_features <- function(file_path) {

  temporary_wav <- tempfile(fileext = ".wav")

  on.exit({
    if (file.exists(temporary_wav)) {
      unlink(temporary_wav)
    }
  }, add = TRUE)

  # Convert audio to mono WAV at the same sample rate used in training.
  av::av_audio_convert(
    audio = file_path,
    output = temporary_wav,
    channels = 1,
    sample_rate = TARGET_SAMPLE_RATE,
    verbose = FALSE
  )

  wave <- tuneR::readWave(temporary_wav)
  signal <- as.numeric(wave@left) / 32768

  if (length(signal) < FRAME_LENGTH) {
    stop(
      "Audio is too short. At least ", FRAME_LENGTH,
      " samples are required after conversion."
    )
  }

  if (any(!is.finite(signal))) {
    stop("The audio contains invalid waveform values.")
  }

  # RMS energy.
  energy <- sqrt(mean(signal^2))

  # Zero-crossing rate.
  zcr_value <- mean(
    abs(diff(sign(signal))) / 2,
    na.rm = TRUE
  )

  # Spectral features use the first 2,048 samples, as in training.
  spectral_features <- calculate_spectral_features(
    signal[1:FRAME_LENGTH],
    TARGET_SAMPLE_RATE
  )

  # MFCC statistics are calculated across the full recording.
  mfcc_features <- calculate_mfcc_features(wave)

  feature_row <- data.frame(
    Pitch_Centroid = unname(spectral_features["Pitch_Centroid"]),
    Energy = energy,
    ZCR = zcr_value,
    Spectral_Bandwidth =
      unname(spectral_features["Spectral_Bandwidth"]),
    Spectral_RollOff85 =
      unname(spectral_features["Spectral_RollOff85"]),
    Spectral_Flatness =
      unname(spectral_features["Spectral_Flatness"]),
    Spectral_Entropy =
      unname(spectral_features["Spectral_Entropy"]),
    as.list(mfcc_features),
    check.names = FALSE
  )

  feature_row <- feature_row[, FEATURE_NAMES, drop = FALSE]

  feature_values <- as.matrix(feature_row)
  storage.mode(feature_values) <- "double"

  if (ncol(feature_values) != 33 ||
      any(!is.finite(feature_values))) {
    stop("Feature extraction failed validation. Check the audio file.")
  }

  feature_row
}

# ------------------------------------------------------------
# 8. PREDICT SPECIES AND THREAT LEVEL
# The SVM models are called directly from model_bundle$species$model
# and model_bundle$threat$model; model_bundle$predict is not required.
# ------------------------------------------------------------

cat("\nExtracting audio features...\n")
new_audio_features <- extract_audio_features(audio_path)

cat("Extracted features :", ncol(new_audio_features), "\n")
cat("Loading SVM models  : completed\n")

# Convert the extracted feature row to a numeric matrix in the exact
# column order used during SVM training.
feature_matrix <- as.matrix(
  new_audio_features[, model_bundle$feature_names, drop = FALSE]
)
storage.mode(feature_matrix) <- "double"

if (nrow(feature_matrix) != 1 ||
    ncol(feature_matrix) != length(model_bundle$feature_names) ||
    any(!is.finite(feature_matrix))) {
  stop("The extracted feature matrix is not valid for prediction.")
}

# Predict directly with the two saved e1071 SVM models.
species_prediction <- stats::predict(
  model_bundle$species$model,
  newdata = feature_matrix
)

threat_prediction <- stats::predict(
  model_bundle$threat$model,
  newdata = feature_matrix
)

prediction <- data.frame(
  Species = as.character(species_prediction),
  Threat_Level = as.character(threat_prediction),
  stringsAsFactors = FALSE
)

if (nrow(prediction) != 1) {
  stop("The models returned an unexpected prediction format.")
}

# ------------------------------------------------------------
# 9. DISPLAY THE FINAL RESULT
# ------------------------------------------------------------

cat("\n")
cat(paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("          WILDLIFE AUDIO CLASSIFICATION RESULT\n")
cat(paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("Recording       : ", basename(audio_path), "\n", sep = "")
cat("Predicted species: ", prediction$Species[1], "\n", sep = "")
cat("Predicted threat : ", prediction$Threat_Level[1], "\n", sep = "")
cat(paste(rep("=", 58), collapse = ""), "\n", sep = "")
cat("Prediction completed successfully.\n")
