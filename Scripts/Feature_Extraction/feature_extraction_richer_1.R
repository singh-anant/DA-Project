# =====================================================================
# Richer Audio Feature Extraction
# Output: Data_Processed/final_ml_features_richer_1.csv
#
# Features extracted for each animal recording:
#   1. Pitch_Centroid (spectral centroid)
#   2. Energy (RMS energy)
#   3. ZCR (zero-crossing rate)
#   4. Spectral_Bandwidth
#   5. Spectral_RollOff85
#   6. Spectral_Flatness
#   7. Spectral_Entropy
#   8. MFCC_1_Mean ... MFCC_13_Mean
#   9. MFCC_1_SD   ... MFCC_13_SD
#
# This script does not overwrite the original feature CSV.
# =====================================================================

library(readr)
library(dplyr)
library(tuneR)
library(seewave)
library(av)

# --------------------------- 1. Paths and settings --------------------

metadata_file <- "Data_Processed/combined_land_mammals.csv"

# Root folder containing the regional animal-audio folders.
audio_root <- "/Volumes/Maniac"

output_file <- "Data_Processed/final_ml_features_richer_1.csv"

TARGET_SAMPLE_RATE <- 22050
FRAME_LENGTH <- 2048
MFCC_COUNT <- 13

dir.create("Data_Processed", showWarnings = FALSE, recursive = TRUE)

if (!file.exists(metadata_file)) {
  stop("Metadata file not found: ", metadata_file)
}

if (!dir.exists(audio_root)) {
  stop("Audio folder not found: ", audio_root,
       "\nCheck whether the external volume is mounted.")
}

# --------------------------- 2. Read metadata -------------------------

metadata <- read_csv(metadata_file, show_col_types = FALSE)

required_metadata_columns <- c("id")
missing_metadata <- setdiff(required_metadata_columns, names(metadata))

if (length(missing_metadata) > 0) {
  stop("Metadata is missing required columns: ",
       paste(missing_metadata, collapse = ", "))
}

metadata <- metadata %>%
  mutate(
    id = as.character(id),
    File_Name = paste0("XC", id, ".mp3")
  ) %>%
  filter(!is.na(id), id != "") %>%
  distinct(File_Name, .keep_all = TRUE)

# --------------------------- 3. Find audio files -----------------------

cat("Searching for audio recordings...\n")

audio_files <- list.files(
  path = audio_root,
  pattern = "\\.mp3$",
  recursive = TRUE,
  full.names = TRUE,
  ignore.case = TRUE
)

audio_lookup <- data.frame(
  File_Name = basename(audio_files),
  Audio_Path = audio_files,
  stringsAsFactors = FALSE
) %>%
  distinct(File_Name, .keep_all = TRUE)

metadata <- metadata %>%
  inner_join(audio_lookup, by = "File_Name")

cat("Recordings found in metadata and audio folders:", nrow(metadata), "\n")

if (nrow(metadata) == 0) {
  stop("No matching XC audio files found. Check audio_root and metadata IDs.")
}

# --------------------------- 4. Helper functions -----------------------

# Calculate spectral features from the power spectrum.
calculate_spectral_features <- function(signal, sample_rate) {

  # Remove any DC offset.
  signal <- signal - mean(signal)

  # Apply a Hann window to reduce spectral leakage.
  window <- 0.5 - 0.5 * cos(
    2 * pi * (0:(length(signal) - 1)) / (length(signal) - 1)
  )

  windowed_signal <- signal * window

  # Calculate FFT and retain the positive-frequency half.
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
    return(c(
      Pitch_Centroid = NA_real_,
      Spectral_Bandwidth = NA_real_,
      Spectral_RollOff85 = NA_real_,
      Spectral_Flatness = NA_real_,
      Spectral_Entropy = NA_real_
    ))
  }

  # Normalize power values to obtain spectral proportions.
  probabilities <- power / total_power

  # Spectral centroid: weighted average frequency.
  centroid <- sum(frequencies * probabilities)

  # Spectral bandwidth: weighted standard deviation around centroid.
  bandwidth <- sqrt(
    sum(probabilities * (frequencies - centroid)^2)
  )

  # Spectral roll-off: frequency below which 85% of power is present.
  cumulative_power <- cumsum(probabilities)
  rolloff_index <- which(cumulative_power >= 0.85)[1]
  rolloff_85 <- frequencies[rolloff_index]

  # Spectral flatness: geometric mean / arithmetic mean of power.
  # A small constant avoids log(0).
  epsilon <- .Machine$double.eps
  flatness <- exp(mean(log(power + epsilon))) /
    mean(power + epsilon)

  # Normalized spectral entropy (range approximately 0 to 1).
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

# Calculate MFCC mean and standard deviation for all 13 coefficients.
calculate_mfcc_features <- function(wave_object) {

  mfcc_values <- tuneR::melfcc(
    wave_object,
    numcep = MFCC_COUNT
  )

  mfcc_matrix <- as.matrix(mfcc_values)

  # Ensure rows represent MFCC coefficients and columns represent frames.
  if (nrow(mfcc_matrix) != MFCC_COUNT &&
      ncol(mfcc_matrix) == MFCC_COUNT) {
    mfcc_matrix <- t(mfcc_matrix)
  }

  if (nrow(mfcc_matrix) != MFCC_COUNT) {
    stop("Unexpected MFCC dimensions returned by melfcc().")
  }

  mfcc_means <- rowMeans(mfcc_matrix, na.rm = TRUE)
  mfcc_sds <- apply(mfcc_matrix, 1, sd, na.rm = TRUE)

  # Replace undefined SD values (for example, a single frame) with zero.
  mfcc_sds[!is.finite(mfcc_sds)] <- 0

  result <- c(mfcc_means, mfcc_sds)

  names(result) <- c(
    paste0("MFCC_", 1:MFCC_COUNT, "_Mean"),
    paste0("MFCC_", 1:MFCC_COUNT, "_SD")
  )

  result
}

# --------------------------- 5. Extract features -----------------------

features_list <- list()
skipped_files <- data.frame(
  File_Name = character(),
  Reason = character(),
  stringsAsFactors = FALSE
)

cat("\nStarting feature extraction...\n")

for (i in seq_len(nrow(metadata))) {

  file_name <- metadata$File_Name[i]
  file_path <- metadata$Audio_Path[i]

  cat(
    "\rProcessing", i, "of", nrow(metadata), ":", file_name,
    "   "
  )

  temp_wav <- tempfile(fileext = ".wav")

  tryCatch({

    # Convert audio to mono WAV at the target sample rate.
    av::av_audio_convert(
      audio = file_path,
      output = temp_wav,
      channels = 1,
      sample_rate = TARGET_SAMPLE_RATE,
      verbose = FALSE
    )

    # Read converted WAV.
    wave <- tuneR::readWave(temp_wav)

    # Extract waveform samples and normalize to [-1, 1].
    signal <- as.numeric(wave@left) / 32768

    if (length(signal) < FRAME_LENGTH) {
      stop("Audio is too short for the selected FFT frame length.")
    }

    if (any(!is.finite(signal))) {
      stop("Audio contains invalid waveform values.")
    }

    # RMS energy.
    energy <- sqrt(mean(signal^2))

    # Zero-crossing rate.
    zcr_value <- mean(
      abs(diff(sign(signal))) / 2,
      na.rm = TRUE
    )

    # Spectral features use the first FRAME_LENGTH samples.
    # This gives one consistent-length spectrum per recording.
    spectral_signal <- signal[1:FRAME_LENGTH]

    spectral_features <- calculate_spectral_features(
      spectral_signal,
      TARGET_SAMPLE_RATE
    )

    # MFCC features are summarized across all available audio frames.
    mfcc_features <- calculate_mfcc_features(wave)

    # Combine all extracted features into one row.
    feature_row <- data.frame(
      File_Name = file_name,
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

    features_list[[length(features_list) + 1]] <- feature_row

  }, error = function(e) {

    skipped_files <<- rbind(
      skipped_files,
      data.frame(
        File_Name = file_name,
        Reason = conditionMessage(e),
        stringsAsFactors = FALSE
      )
    )

  }, finally = {

    if (file.exists(temp_wav)) {
      unlink(temp_wav)
    }

  })
}

cat("\n\nFeature extraction finished.\n")

# --------------------------- 6. Save results ---------------------------

if (length(features_list) == 0) {
  stop("No features were extracted. Check the skipped-files report.")
}

final_features <- bind_rows(features_list)

# Keep File_Name first, followed by numerical feature columns.
final_features <- final_features %>%
  select(File_Name, everything())

write_csv(final_features, output_file)

cat("Output file:", output_file, "\n")
cat("Successfully processed recordings:", nrow(final_features), "\n")
cat("Number of columns:", ncol(final_features), "\n")
cat("Number of numerical audio features:", ncol(final_features) - 1, "\n")

# Save skipped recordings for troubleshooting.
if (nrow(skipped_files) > 0) {

  skipped_file <- "Data_Processed/richer_1_skipped_files.csv"
  write_csv(skipped_files, skipped_file)

  cat("Skipped recordings:", nrow(skipped_files), "\n")
  cat("Skipped-file report:", skipped_file, "\n")

} else {

  cat("No recordings were skipped.\n")

}

cat("\nFirst few extracted rows:\n")
print(head(final_features))
