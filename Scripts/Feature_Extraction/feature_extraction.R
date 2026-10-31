# Import required libraries for feature extraction
library(dplyr)
library(readr)
library(tuneR)
library(seewave)
library(av)

# Load the combined CSV file
df <- read_csv("Data_Processed/combined_land_mammals.csv", show_col_types = FALSE)

# Define the pendrive folder paths
pendrive_base <- "/Volumes/Maniac"
audio_folders <- c(
  file.path(pendrive_base, "Africa Land Mammels"),
  file.path(pendrive_base, "America Land Mammels"),
  file.path(pendrive_base, "Asia Land Mammels"),
  file.path(pendrive_base, "Australia Land Mammels"),
  file.path(pendrive_base, "Europe Land Mammels")
)

# Create an empty list for the extracted features
features_list <- list()
cat("Starting safe feature extraction on Maniac Pendrive for", nrow(df), "files...\n")

# Loop through each row in the dataframe to extract features
for (i in 1:nrow(df)) {
  # Generate the expected audio file name based on the id column
  file_name <- paste0("XC", df$id[i], ".mp3")

  # Search all 5 pendrive folders for this specific audio file
  file_path <- ""

  # Loop through each folder to find the file
  for (folder in audio_folders) {
    # Construct the full path to the audio file
    temp_path <- file.path(folder, file_name)

    # Check if the file exists in this folder
    if (file.exists(temp_path)) {
      file_path <- temp_path
      break
    }
  }

  # If the file path is still empty, skip to the next iteration
  if (file_path == "") next

  tryCatch({
    # Convert the MP3 to a temporary WAV file, then read the WAV
    #WAV is used because it is generally easier to process for audio analysis.
    temp_wav <- tempfile(fileext = ".wav")
    av_audio_convert(file_path, temp_wav, verbose = FALSE)
    
    #reads the converted WAV audio file into R.
    wave <- readWave(temp_wav)

    # Delete the temporary wav file immediately to save space
    unlink(temp_wav)

    # Calculate the spectral centroid and energy
    spec_data <- meanspec(wave, plot = FALSE)
    centroid <- specprop(spec_data)$cent
    energy <- sqrt(mean(wave@left^2))

    # Calculate Zero Crossing Rate (ZCR)
    zcr_vals <- zcr(wave, wl = 512, plot = FALSE)

    # Calculate the mean ZCR across all frames
    mean_zcr <- mean(zcr_vals[, 2], na.rm = TRUE)

    # Calculate MFCCs
    mfcc_matrix <- melfcc(wave, numcep = 13)

    # Calculate the mean of each MFCC coefficient across all frames
    mfcc_means <- colMeans(mfcc_matrix, na.rm = TRUE)

    # Store the extracted features
    row_data <- data.frame(
      File_Name = file_name,
      Pitch_Centroid = centroid,
      Energy = energy,
      ZCR = mean_zcr
    )

    # t takes 13 MFCC values calculated from an audio recording and saves them as 13 separate numerical features for the machine-learning model.
    for (m in 1:13) {
      row_data[[paste0("MFCC_", m)]] <- mfcc_means[m]
    }

    features_list[[i]] <- row_data
    cat("Processed [", i, "/", nrow(df), "]:", file_name, "\n")

  }, error = function(e) {
    cat("Skipped:", file_name, "- Error extracting features\n")
  })
}

# Export the final matrix for Random Forest
final_ml_df <- bind_rows(features_list)
write_csv(final_ml_df, "Data_Processed/final_ml_features.csv")
cat("\n✅ SUCCESS! Saved to Data_Processed/final_ml_features.csv\n")