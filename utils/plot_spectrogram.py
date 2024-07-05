import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_mel_spectrogram(mel_spectrogram:np.ndarray, sr:int=22050, fmax:int=8000):
  """
  Plots a Mel spectrogram

  Parameters:
  - mel_spectrogram (np.ndarray): The Mel spectrogram
  - sr (int): The sample rate
  - fmax (int): Maximum frequency in Hz
  """
  plt.figure(figsize=(10, 4))
  librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', sr=sr, fmax=fmax)
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel spectrogram')
  plt.tight_layout()
  plt.show()