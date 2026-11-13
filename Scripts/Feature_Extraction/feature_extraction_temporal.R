# =====================================================================
# Temporal + Spectral Feature Extraction for Wild-Animal Audio
#
# Keeps the original 33 richer features and adds 40 temporal features:
#   1. Frame-level summaries (mean + SD) for 7 descriptors = 14
#   2. First-difference MFCC summaries (mean + SD for 13 MFCCs) = 26
#
# Total: 73 numerical audio predictors + File_Name.
#
# Input : Data_Processed/combined_land_mammals.csv
#         Audio files under /Volumes/Maniac/<regional folders>
# Output: Data_Processed/final_ml_features_richer_temporal.csv
# Skip log: Data_Processed/feature_extraction_temporal_skipped_files.csv
#
# This script does NOT overwrite any existing feature dataset.
# =====================================================================

library(readr)
library(dplyr)
library(tuneR)
library(seewave)
library(av)

# --------------------------- 1. Settings ------------------------------
metadata_file <- "Data_Processed/combined_land_mammals.csv"
output_file <- "Data_Processed/final_ml_features_richer_temporal.csv"
skipped_file <- "Data_Processed/feature_extraction_temporal_skipped_files.csv"

pendrive_base <- "/Volumes/Maniac"
audio_folders <- c(
  file.path(pendrive_base, "Africa Land Mammels"),
  file.path(pendrive_base, "America Land Mammels"),
  file.path(pendrive_base, "Asia Land Mammels"),
  file.path(pendrive_base, "Australia Land Mammels"),
  file.path(pendrive_base, "Europe Land Mammels")
)

TARGET_SAMPLE_RATE <- 22050
FRAME_LENGTH <- 1024
HOP_LENGTH <- 512
ROLLOFF_PERCENT <- 0.85

if (!file.exists(metadata_file)) {
  stop("Metadata CSV not found: ", metadata_file)
}

df <- read_csv(metadata_file, show_col_types = FALSE)

if (!"id" %in% names(df)) {
  stop("Metadata must contain an 'id' column.")
}

# --------------------------- 2. Locate audio ---------------------------
find_audio_file <- function(file_name, folders) {
  for (folder in folders) {
    candidate <- file.path(folder, file_name)
    if (file.exists(candidate)) return(candidate)
  }
  NA_character_
}

# --------------------------- 3. Overall spectral features --------------
# These calculations preserve the approach used in the original
# feature_extraction_richer.R script.
calculate_spectral_features <- function(samples, sample_rate,
                                         frame_length = 1024,
                                         hop_length = 512,
                                         rolloff_percent = 0.85) {
  x <- as.numeric(samples)
  x <- x[is.finite(x)]

  if (!length(x)) stop("No finite audio samples.")

  x <- x - mean(x)

  if (length(x) < frame_length) {
    x <- c(x, rep(0, frame_length - length(x)))
  }

  starts <- seq(1, length(x) - frame_length + 1, by = hop_length)
  hann <- 0.5 - 0.5 * cos(2 * pi * (0:(frame_length - 1)) / (frame_length - 1))
  n_bins <- frame_length %/% 2 + 1
  power_sum <- numeric(n_bins)

  for (start in starts) {
    idx <- start:(start + frame_length - 1)
    frame <- x[idx] * hann
    fft_values <- fft(frame)
    power_sum <- power_sum + Mod(fft_values[seq_len(n_bins)])^2
  }

  mean_power <- power_sum / length(starts)
  freq_hz <- (0:(n_bins - 1)) * sample_rate / frame_length
  total_power <- sum(mean_power)

  if (!is.finite(total_power) || total_power <= 0) {
    stop("Audio is silent or has no measurable spectral power.")
  }

  magnitude <- sqrt(mean_power)
  total_magnitude <- sum(magnitude)
  centroid_hz <- sum(freq_hz * magnitude) / total_magnitude
  bandwidth_hz <- sqrt(
    sum(magnitude * (freq_hz - centroid_hz)^2) / total_magnitude
  )

  cumulative_power <- cumsum(mean_power) / total_power
  rolloff_index <- which(cumulative_power >= rolloff_percent)[1]
  rolloff_hz <- freq_hz[rolloff_index]

  epsilon <- max(mean_power) * 1e-12 + .Machine$double.eps
  p_for_log <- mean_power + epsilon
  flatness <- exp(mean(log(p_for_log))) / mean(p_for_log)

  probability <- mean_power / total_power
  p_positive <- probability[probability > 0]
  entropy <- -sum(p_positive * log2(p_positive)) / log2(length(probability))

  data.frame(
    Spectral_Centroid = as.numeric(centroid_hz),
    Spectral_Bandwidth = as.numeric(bandwidth_hz),
    Spectral_RollOff85 = as.numeric(rolloff_hz),
    Spectral_Flatness = as.numeric(flatness),
    Spectral_Entropy = as.numeric(entropy)
  )
}

# --------------------------- 4. Frame-level temporal features ---------
# Calculate acoustic descriptors separately for successive overlapping
# frames. Each frame is 1024 samples (~46 ms); hop is 512 samples (~23 ms).
calculate_frame_descriptors <- function(samples, sample_rate,
                                         frame_length = 1024,
                                         hop_length = 512,
                                         rolloff_percent = 0.85) {
  x <- as.numeric(samples)
  x <- x[is.finite(x)]

  if (!length(x)) stop("No finite audio samples for frame features.")

  if (length(x) < frame_length) {
    x <- c(x, rep(0, frame_length - length(x)))
  }

  starts <- seq(1, length(x) - frame_length + 1, by = hop_length)
  hann <- 0.5 - 0.5 * cos(2 * pi * (0:(frame_length - 1)) / (frame_length - 1))
  n_bins <- frame_length %/% 2 + 1
  freq_hz <- (0:(n_bins - 1)) * sample_rate / frame_length

  frame_results <- vector("list", length(starts))

  for (j in seq_along(starts)) {
    start <- starts[j]
    idx <- start:(start + frame_length - 1)
    raw_frame <- x[idx]

    # Time-domain descriptors use the unwindowed frame.
    frame_rms <- sqrt(mean(raw_frame^2))
    frame_zcr <- sum(diff(sign(raw_frame)) != 0) / (2 * (length(raw_frame) - 1))

    # Center and window the frame before its Fourier transform.
    centered_frame <- raw_frame - mean(raw_frame)
    windowed_frame <- centered_frame * hann
    fft_values <- fft(windowed_frame)
    power <- Mod(fft_values[seq_len(n_bins)])^2
    total_power <- sum(power)

    if (!is.finite(total_power) || total_power <= 0) {
      stop("A frame is silent or has no measurable spectral power.")
    }

    magnitude <- sqrt(power)
    total_magnitude <- sum(magnitude)

    centroid <- sum(freq_hz * magnitude) / total_magnitude
    bandwidth <- sqrt(
      sum(magnitude * (freq_hz - centroid)^2) / total_magnitude
    )

    cumulative_power <- cumsum(power) / total_power
    rolloff_index <- which(cumulative_power >= rolloff_percent)[1]
    rolloff <- freq_hz[rolloff_index]

    epsilon <- max(power) * 1e-12 + .Machine$double.eps
    flatness <- exp(mean(log(power + epsilon))) / mean(power + epsilon)

    probability <- power / total_power
    p_positive <- probability[probability > 0]
    entropy <- -sum(p_positive * log2(p_positive)) / log2(length(probability))

    frame_results[[j]] <- c(
      Frame_RMS = frame_rms,
      Frame_ZCR = frame_zcr,
      Frame_Spectral_Centroid = centroid,
      Frame_Spectral_Bandwidth = bandwidth,
      Frame_Spectral_RollOff85 = rolloff,
      Frame_Spectral_Flatness = flatness,
      Frame_Spectral_Entropy = entropy
    )
  }

  result <- as.data.frame(do.call(rbind, frame_results))
  return(result)
}

# Summarize a time series using its mean and standard deviation.
# If only one frame exists, SD is set to zero instead of NA.
summarize_temporal_series <- function(values, prefix) {
  values <- as.numeric(values)
  values <- values[is.finite(values)]

  if (!length(values)) stop("No finite values for: ", prefix)

  value_sd <- if (length(values) > 1) sd(values) else 0
  if (!is.finite(value_sd)) value_sd <- 0

  result <- c(mean(values), value_sd)
  names(result) <- c(
    paste0(prefix, "_Mean"),
    paste0(prefix, "_SD")
  )
  return(result)
}

# --------------------------- 5. Extract features -----------------------
features_list <- list()
skipped_list <- list()

cat("Starting temporal feature extraction for", nrow(df), "metadata rows.\n")
cat("Output:", output_file, "\n")

for (i in seq_len(nrow(df))) {

  recording_id <- df$id[i]

  if (is.na(recording_id) || trimws(as.character(recording_id)) == "") {
    skipped_list[[length(skipped_list) + 1]] <- data.frame(
      File_Name = NA_character_,
      Reason = "Missing recording ID"
    )
    next
  }

  file_name <- paste0("XC", recording_id, ".mp3")
  file_path <- find_audio_file(file_name, audio_folders)

  if (is.na(file_path)) {
    skipped_list[[length(skipped_list) + 1]] <- data.frame(
      File_Name = file_name,
      Reason = "Audio file not found in configured folders"
    )
    next
  }

  temp_wav <- tempfile(fileext = ".wav")

  tryCatch({

    # Standardize recordings to mono and 22050 Hz.
    av_audio_convert(
      audio = file_path,
      output = temp_wav,
      channels = 1,
      sample_rate = TARGET_SAMPLE_RATE,
      verbose = FALSE
    )

    wave <- tuneR::readWave(temp_wav)
    samples <- as.numeric(wave@left)
    sample_rate <- wave@samp.rate

    if (!length(samples)) stop("Empty audio recording.")
    if (sample_rate != TARGET_SAMPLE_RATE) {
      stop("Unexpected sample rate: ", sample_rate)
    }

    # ================================================================
    # A. Original 33 features (kept compatible with previous pipeline)
    # ================================================================

    energy <- sqrt(mean(samples^2, na.rm = TRUE))

    zcr_values <- seewave::zcr(
      wave,
      f = sample_rate,
      wl = FRAME_LENGTH,
      plot = FALSE
    )
    mean_zcr <- mean(zcr_values[, 2], na.rm = TRUE)

    spectral <- calculate_spectral_features(
      samples,
      sample_rate,
      FRAME_LENGTH,
      HOP_LENGTH,
      ROLLOFF_PERCENT
    )

    mfcc_matrix <- tuneR::melfcc(
      wave,
      sr = sample_rate,
      numcep = 13,
      wintime = 0.025,
      hoptime = 0.010,
      frames_in_rows = TRUE
    )

    if (ncol(mfcc_matrix) != 13) {
      stop("Expected 13 MFCC columns; received ", ncol(mfcc_matrix), ".")
    }

    mfcc_means <- colMeans(mfcc_matrix, na.rm = TRUE)
    mfcc_sds <- apply(mfcc_matrix, 2, sd, na.rm = TRUE)
    mfcc_sds[!is.finite(mfcc_sds)] <- 0

    # ================================================================
    # B. 14 temporal summaries from seven frame-level descriptors
    # ================================================================

    frame_data <- calculate_frame_descriptors(
      samples,
      sample_rate,
      FRAME_LENGTH,
      HOP_LENGTH,
      ROLLOFF_PERCENT
    )

    temporal_values <- numeric(0)

    frame_prefixes <- c(
      "Frame_RMS",
      "Frame_ZCR",
      "Frame_Spectral_Centroid",
      "Frame_Spectral_Bandwidth",
      "Frame_Spectral_RollOff85",
      "Frame_Spectral_Flatness",
      "Frame_Spectral_Entropy"
    )

    for (prefix in frame_prefixes) {
      temporal_values <- c(
        temporal_values,
        summarize_temporal_series(frame_data[[prefix]], prefix)
      )
    }

    # ================================================================
    # C. 26 MFCC first-difference (delta-like) temporal summaries
    # ================================================================

    # First differences describe frame-to-frame changes in each MFCC.
    # These are simple first differences, not regression-based MFCC deltas.
    mfcc_delta_means <- numeric(13)
    mfcc_delta_sds <- numeric(13)

    if (nrow(mfcc_matrix) >= 2) {
      for (m in 1:13) {
        delta_values <- diff(mfcc_matrix[, m])
        delta_values <- delta_values[is.finite(delta_values)]

        mfcc_delta_means[m] <- if (length(delta_values)) mean(delta_values) else 0
        mfcc_delta_sds[m] <- if (length(delta_values) > 1) sd(delta_values) else 0
      }
    } else {
      mfcc_delta_means[] <- 0
      mfcc_delta_sds[] <- 0
    }

    names(mfcc_delta_means) <- paste0("MFCC_", 1:13, "_Delta_Mean")
    names(mfcc_delta_sds) <- paste0("MFCC_", 1:13, "_Delta_SD")

    # ================================================================
    # D. Combine all 73 numerical features
    # ================================================================

    row_data <- data.frame(
      File_Name = file_name,
      # This is spectral centroid (Hz), retained under the old column name.
      Pitch_Centroid = as.numeric(spectral$Spectral_Centroid),
      Energy = as.numeric(energy),
      ZCR = as.numeric(mean_zcr),
      Spectral_Bandwidth = as.numeric(spectral$Spectral_Bandwidth),
      Spectral_RollOff85 = as.numeric(spectral$Spectral_RollOff85),
      Spectral_Flatness = as.numeric(spectral$Spectral_Flatness),
      Spectral_Entropy = as.numeric(spectral$Spectral_Entropy)
    )

    for (m in 1:13) {
      row_data[[paste0("MFCC_", m, "_Mean")]] <- as.numeric(mfcc_means[m])
    }

    for (m in 1:13) {
      row_data[[paste0("MFCC_", m, "_SD")]] <- as.numeric(mfcc_sds[m])
    }

    for (feature_name in names(temporal_values)) {
      row_data[[feature_name]] <- as.numeric(temporal_values[feature_name])
    }

    for (m in 1:13) {
      row_data[[paste0("MFCC_", m, "_Delta_Mean")]] <- as.numeric(mfcc_delta_means[m])
    }

    for (m in 1:13) {
      row_data[[paste0("MFCC_", m, "_Delta_SD")]] <- as.numeric(mfcc_delta_sds[m])
    }

    numeric_values <- unlist(row_data[setdiff(names(row_data), "File_Name")])

    if (length(numeric_values) != 73) {
      stop("Expected 73 numeric features; extracted ", length(numeric_values), ".")
    }

    if (any(!is.finite(numeric_values))) {
      stop("Feature contains NA, NaN, or infinite value.")
    }

    features_list[[length(features_list) + 1]] <- row_data

    cat(
      "Processed [", i, "/", nrow(df), "]: ", file_name, "\n",
      sep = ""
    )

  }, error = function(e) {

    skipped_list[[length(skipped_list) + 1]] <<- data.frame(
      File_Name = file_name,
      Reason = conditionMessage(e)
    )

    cat(
      "Skipped: ", file_name, " - ", conditionMessage(e), "\n",
      sep = ""
    )

  }, finally = {
    if (file.exists(temp_wav)) unlink(temp_wav)
  })
}

# --------------------------- 6. Save outputs ---------------------------
if (!length(features_list)) {
  stop("No features were extracted. Check drive mount, paths and extraction errors.")
}

final_ml_df <- bind_rows(features_list)

if (anyDuplicated(final_ml_df$File_Name) > 0) {
  stop("Duplicate File_Name values detected.")
}

if (ncol(final_ml_df) != 74) {
  stop("Expected 74 columns including File_Name; found ", ncol(final_ml_df), ".")
}

write_csv(final_ml_df, output_file, na = "")

cat("\nTEMPORAL FEATURE EXTRACTION COMPLETE\n")
cat("Successfully processed:", nrow(final_ml_df), "recordings\n")
cat("Numerical audio features:", ncol(final_ml_df) - 1, "\n")
cat("Saved:", output_file, "\n")

if (length(skipped_list)) {
  skipped_df <- bind_rows(skipped_list)
  write_csv(skipped_df, skipped_file, na = "")
  cat("Skipped recordings:", nrow(skipped_df), "\n")
  cat("Skip log:", skipped_file, "\n")
}
