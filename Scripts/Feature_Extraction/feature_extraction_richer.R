# ============================================================
# Improved Feature Extraction for Wild Animal Audio
# Part 1: Libraries, paths and helper functions
# ============================================================

library(dplyr)
library(readr)
library(tuneR)
library(seewave)
library(av)

# -------------------- 1. File paths -------------------------

# Input metadata file
metadata_file <- "Data_Processed/combined_land_mammals.csv"

# Save to a NEW file. Do not overwrite the original features.
output_file <- "Data_Processed/final_ml_features_richer.csv"

# Audio folders on the external drive
pendrive_base <- "/Volumes/Maniac"

audio_folders <- c(
  file.path(pendrive_base, "Africa Land Mammels"),
  file.path(pendrive_base, "America Land Mammels"),
  file.path(pendrive_base, "Asia Land Mammels"),
  file.path(pendrive_base, "Australia Land Mammels"),
  file.path(pendrive_base, "Europe Land Mammels")
)

# Audio processing settings
TARGET_SAMPLE_RATE <- 22050
FRAME_LENGTH <- 1024
HOP_LENGTH <- 512

# Check that metadata exists
if (!file.exists(metadata_file)) {
  stop("Metadata file not found: ", metadata_file)
}

# Read metadata
df <- read_csv(
  metadata_file,
  show_col_types = FALSE
)

if (!"id" %in% names(df)) {
  stop("The metadata CSV must contain an id column.")
}

# -------------------- 2. Find audio file --------------------

find_audio_file <- function(file_name, folders) {

  for (folder in folders) {

    candidate <- file.path(folder, file_name)

    if (file.exists(candidate)) {
      return(candidate)
    }
  }

  return(NA_character_)
}

# -------------------- 3. Spectral features ------------------

# Calculate the spectral centroid, bandwidth, roll-off,
# flatness and normalized spectral entropy.

calculate_spectral_features <- function(
    samples,
    sample_rate,
    frame_length = 1024,
    hop_length = 512) {

  # Convert samples to numeric and remove invalid values.
  x <- as.numeric(samples)
  x <- x[is.finite(x)]

  if (length(x) == 0) {
    stop("No valid audio samples.")
  }

  # Remove DC offset.
  x <- x - mean(x)

  # Pad short recordings to fit at least one frame.
  if (length(x) < frame_length) {
    x <- c(
      x,
      rep(0, frame_length - length(x))
    )
  }

  # Find the starting position of each frame.
  starts <- seq(
    1,
    length(x) - frame_length + 1,
    by = hop_length
  )

  # Create a Hann window.
  window <- 0.5 - 0.5 * cos(
    2 * pi * (0:(frame_length - 1)) /
      (frame_length - 1)
  )

  # Number of non-negative frequency bins.
  number_of_bins <- frame_length %/% 2 + 1

  # Store average power across frames.
  power_sum <- numeric(number_of_bins)

  for (start in starts) {

    indices <- start:(start + frame_length - 1)

    frame <- x[indices] * window

    fft_values <- fft(frame)

    # Retain non-negative frequencies.
    spectrum <- fft_values[
      seq_len(number_of_bins)
    ]

    frame_power <- Mod(spectrum)^2

    power_sum <- power_sum + frame_power
  }

  # Average power spectrum.
  mean_power <- power_sum / length(starts)

  frequencies <- (
    0:(number_of_bins - 1)
  ) * sample_rate / frame_length

  total_power <- sum(mean_power)

  if (!is.finite(total_power) || total_power <= 0) {
    stop("Audio is silent or has no measurable power.")
  }

  # Use the square root of power as the magnitude spectrum.
  magnitude <- sqrt(mean_power)

  total_magnitude <- sum(magnitude)

  # Spectral centroid in Hz.
  centroid <- sum(
    frequencies * magnitude
  ) / total_magnitude

  # Spectral bandwidth in Hz.
  bandwidth <- sqrt(
    sum(
      magnitude * (frequencies - centroid)^2
    ) / total_magnitude
  )

  # Spectral roll-off at 85% of cumulative power.
  cumulative_power <- cumsum(mean_power) / total_power

  rolloff_index <- which(
    cumulative_power >= 0.85
  )[1]

  rolloff <- frequencies[rolloff_index]

  # Spectral flatness: geometric mean / arithmetic mean.
  epsilon <- max(mean_power) * 1e-12 +
    .Machine$double.eps

  safe_power <- mean_power + epsilon

  flatness <- exp(
    mean(log(safe_power))
  ) / mean(safe_power)

  # Normalized spectral entropy.
  probability <- mean_power / total_power

  probability <- probability[probability > 0]

  entropy <- -sum(
    probability * log2(probability)
  ) / log2(number_of_bins)

  return(
    data.frame(
      Pitch_Centroid = centroid,
      Spectral_Bandwidth = bandwidth,
      Spectral_RollOff85 = rolloff,
      Spectral_Flatness = flatness,
      Spectral_Entropy = entropy
    )
  )
}
# ============================================================
# Part 2: Process audio and extract all 33 features
# ============================================================

features_list <- list()
skipped_files <- list()

cat(
  "Starting improved feature extraction for",
  nrow(df), "recordings...\n"
)

for (i in seq_len(nrow(df))) {

  # Generate audio filename from metadata ID.
  recording_id <- df$id[i]

  if (
    is.na(recording_id) ||
    trimws(as.character(recording_id)) == ""
  ) {

    skipped_files[[length(skipped_files) + 1]] <-
      data.frame(
        File_Name = NA_character_,
        Reason = "Missing recording ID"
      )

    next
  }

  file_name <- paste0("XC", recording_id, ".mp3")

  # Find the audio in one of the five regional folders.
  file_path <- find_audio_file(
    file_name,
    audio_folders
  )

  if (is.na(file_path)) {

    skipped_files[[length(skipped_files) + 1]] <-
      data.frame(
        File_Name = file_name,
        Reason = "Audio file not found"
      )

    next
  }

  # Temporary WAV file for processing.
  temp_wav <- tempfile(fileext = ".wav")

  # Process one recording safely.
  result <- tryCatch({

    # Convert audio to mono and 22050 Hz.
    av_audio_convert(
      audio = file_path,
      output = temp_wav,
      channels = 1,
      sample_rate = TARGET_SAMPLE_RATE,
      verbose = FALSE
    )

    # Read standardized audio.
    wave <- tuneR::readWave(temp_wav)

    samples <- as.numeric(wave@left)
    sample_rate <- wave@samp.rate

    if (
      length(samples) == 0 ||
      sample_rate != TARGET_SAMPLE_RATE
    ) {
      stop("Invalid audio or unexpected sample rate.")
    }

    # --------------------------------------------------------
    # A. Spectral features
    # --------------------------------------------------------

    spectral_features <- calculate_spectral_features(
      samples = samples,
      sample_rate = sample_rate,
      frame_length = FRAME_LENGTH,
      hop_length = HOP_LENGTH
    )

    # --------------------------------------------------------
    # B. RMS energy
    # --------------------------------------------------------

    energy <- sqrt(
      mean(samples^2, na.rm = TRUE)
    )

    # --------------------------------------------------------
    # C. Zero-crossing rate
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # D. Calculate 13 MFCCs
    # --------------------------------------------------------

    mfcc_matrix <- tuneR::melfcc(
      wave,
      sr = sample_rate,
      numcep = 13,
      wintime = 0.025,
      hoptime = 0.010,
      frames_in_rows = TRUE
    )

    # With frames_in_rows = TRUE, expect:
    # rows = time frames, columns = 13 MFCC coefficients.

    if (ncol(mfcc_matrix) != 13) {
      stop("MFCC extraction did not return 13 columns.")
    }

    # Average MFCC values across time.
    mfcc_means <- colMeans(
      mfcc_matrix,
      na.rm = TRUE
    )

    # Measure variation in each MFCC over time.
    mfcc_sds <- apply(
      mfcc_matrix,
      2,
      sd,
      na.rm = TRUE
    )

    # If only one valid frame exists, SD can be NA.
    mfcc_sds[!is.finite(mfcc_sds)] <- 0

    # --------------------------------------------------------
    # E. Combine all features into a single row
    # --------------------------------------------------------

    row_data <- data.frame(
      File_Name = file_name,

      # Original acoustic features
      Pitch_Centroid = spectral_features$Pitch_Centroid,
      Energy = energy,
      ZCR = mean_zcr,

      # Additional spectral features
      Spectral_Bandwidth =
        spectral_features$Spectral_Bandwidth,

      Spectral_RollOff85 =
        spectral_features$Spectral_RollOff85,

      Spectral_Flatness =
        spectral_features$Spectral_Flatness,

      Spectral_Entropy =
        spectral_features$Spectral_Entropy
    )

    # Add 13 MFCC mean features.
    for (m in 1:13) {

      row_data[[paste0("MFCC_", m, "_Mean")]] <-
        mfcc_means[m]
    }

    # Add 13 MFCC standard deviation features.
    for (m in 1:13) {

      row_data[[paste0("MFCC_", m, "_SD")]] <-
        mfcc_sds[m]
    }

    # Validate every numeric feature.
    numeric_features <- unlist(
      row_data[, -1, drop = FALSE]
    )

    if (any(!is.finite(numeric_features))) {
      stop("Features contain NA, NaN, or infinite values.")
    }

    # Return the completed feature row.
    list(
      success = TRUE,
      data = row_data
    )

  }, error = function(e) {

    # Record the failure rather than stopping the full loop.
    list(
      success = FALSE,
      reason = conditionMessage(e)
    )

  }, finally = {

    # Always delete the temporary WAV file.
    if (file.exists(temp_wav)) {
      unlink(temp_wav)
    }
  })

  # Store successful rows or log skipped recordings.
  if (result$success) {

    features_list[[length(features_list) + 1]] <-
      result$data

    cat(
      "Processed [", i, "/", nrow(df), "]: ",
      file_name, "\n",
      sep = ""
    )

  } else {

    skipped_files[[length(skipped_files) + 1]] <-
      data.frame(
        File_Name = file_name,
        Reason = result$reason
      )

    cat(
      "Skipped: ", file_name, " - ",
      result$reason, "\n",
      sep = ""
    )
  }
}
# ============================================================
# Part 3: Validate and save extracted features
# ============================================================

# Stop if no recording was successfully processed.
if (length(features_list) == 0) {
  stop(
    "No features were extracted. ",
    "Check the audio paths and extraction errors."
  )
}

# Combine all successful feature rows.
final_ml_df <- bind_rows(features_list)

# Check that recording filenames are unique.
if (anyDuplicated(final_ml_df$File_Name) > 0) {
  stop(
    "Duplicate audio filenames found. ",
    "Check the metadata before continuing."
  )
}

# Save the new feature dataset.
write_csv(
  final_ml_df,
  output_file,
  na = ""
)

# Save skipped recordings for troubleshooting.
if (length(skipped_files) > 0) {

  skipped_df <- bind_rows(skipped_files)

  write_csv(
    skipped_df,
    "Data_Processed/feature_extraction_skipped_files.csv",
    na = ""
  )

  cat(
    "\nSkipped recordings:",
    nrow(skipped_df), "\n"
  )

  cat(
    "Skip log saved to: ",
    "Data_Processed/feature_extraction_skipped_files.csv\n",
    sep = ""
  )
}

# Print a clean summary.
cat("\n====================================\n")
cat("FEATURE EXTRACTION COMPLETE\n")
cat("====================================\n")

cat(
  "Successfully processed:",
  nrow(final_ml_df), "recordings\n"
)

cat(
  "Numerical audio features:",
  ncol(final_ml_df) - 1, "\n"
)

cat(
  "Total columns, including File_Name:",
  ncol(final_ml_df), "\n"
)

cat("Output saved to:", output_file, "\n")