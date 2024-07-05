import librosa
import numpy as np

def wav_to_mel_spectrogram(file_path:str, n_mels:int=128, fmax:int = 8000) -> tuple:
  """ 
  Converts a wav file to a Mel spectrogram

  Parameters:
  - file_path (str): The path to the wav file
  - n_mels (int): The number of Mel bands to generate
  - fmax (int): Maximum frequency in Hz

  Returns:
  - np.ndarray: The Mel spectrogram
  """
  # Load the audio file
  y, sr = librosa.load(file_path, sr=None)

  # Compute the Mel spectrogram
  mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax)

  # Convert the Mel spectrogram to a log scale (dB)
  mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

  return mel_spectrogram, sr