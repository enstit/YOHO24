import librosa
import numpy as np

def wav_to_mel_spectrogram(file_path:str, 
                           n_mels:int = 64, 
                           hop_lenght:int = 10,
                           win_length:int = 25,
                           sr: int = 16000,
                           fmin:int = 0,
                           fmax:int = 7500) -> tuple:
  """ 
  Converts a wav file to a Mel spectrogram

  Parameters:
  - file_path (str): The path to the audio file (WAV or MP3).
  - n_mels (int): The number of Mel bands to generate.
  - fmin (int): Minimum frequency in Hz.
  - fmax (int): Maximum frequency in Hz.
  - sr (int): Sampling rate for the audio file.
  - hop_length (int): Number of samples between successive frames (hop size).
  - win_length (int): Size of the FFT window (window size).
    
  Returns:
  - np.ndarray: The Mel spectrogram.
  - int: The sample rate of the audio file.
  """
  # Load the audio file
  y, origin_sr = librosa.load(file_path, sr=None)

  # Resample the target sample rate
  if origin_sr != sr:
    y = librosa.resample(y=y, orig_sr=origin_sr, target_sr=sr)

  # Convert to mono by averaging channels (if needed)
  if y.ndim > 1:
    y = np.mean(y, axis=0)

  # Compute the Mel spectrogram
  mel_spectrogram = librosa.feature.melspectrogram(
    y=y, 
    sr=sr, 
    n_mels=n_mels, 
    fmin=fmin, 
    fmax=fmax,
    hop_length=hop_lenght, 
    win_length=win_length
  )

  # Convert the Mel spectrogram to a log scale (dB)
  mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

  return mel_spectrogram, sr