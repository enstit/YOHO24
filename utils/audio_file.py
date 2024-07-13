import torch
import librosa
import numpy as np
from matplotlib import pyplot as plt


class AudioFile:
    def __init__(self, file_path: str, labels: list = None):
        """
        Initializes the AudioFile class.

        Args:
            file_path (str): Path to the audio file.
        """
        self.file_path = file_path
        self.labels = labels

    @property
    def mel_spectrogram(self):
        return MelSpectrogram(file_path=self.file_path)


class MelSpectrogram:

    def __init__(self, file_path: str, n_mels: int = 64, hop_lenght: int = 10, win_length: int = 25, sr: int = 16000, fmin: int = 0, fmax: int = 7500):
        """
        Initializes the MelSpectrogram class.

        Args:
            n_mels (int): The number of Mel bands to generate.
            fmin (int): Minimum frequency in Hz.
            fmax (int): Maximum frequency in Hz.
            sr (int): Sampling rate for the audio file.
            hop_length (int): Number of samples between successive frames (hop size).
            win_length (int): Size of the FFT window (window size).
        """
        self.file_path = file_path
        self.n_mels = n_mels
        self.hop_lenght = hop_lenght
        self.win_length = win_length
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.mel_spectrogram = self.compute()
        self.is_normalized = False

    def compute(self, mono: bool = True, normalize: bool = False) -> np.ndarray:
        """
        Converts a wav file to a Mel spectrogram.

        Input:
        - mono (bool): Whether to convert the audio to mono.

        Returns:
        - np.ndarray: The Mel spectrogram.
        """

        # Load the audio file using the native sampling rate
        y, origin_sr = librosa.load(self.file_path, sr=None)

        # Resample the target sample rate
        if origin_sr != self.sr:
            y = librosa.resample(y=y, orig_sr=origin_sr, target_sr=self.sr)

        # Convert to mono by averaging channels (if needed)
        if mono is True and y.ndim > 1:
            y = np.mean(y, axis=0)

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_lenght,
            win_length=self.win_length
        )

        # Convert the Mel spectrogram to a log scale (dB)
        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram, ref=np.max)

        self.mel_spectrogram = mel_spectrogram

        if normalize is True:
            self.normalize()

        return self.mel_spectrogram

    def normalize(self):
        """
        Normalizes the Mel spectrogram.
        """
        if self.is_normalized:
            return
        self.mel_spectrogram = (
            self.mel_spectrogram - np.mean(self.mel_spectrogram)) / np.std(self.mel_spectrogram)
        self.is_normalized = True

    def plot(self):
        """
        Plots the Mel spectrogram.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            self.mel_spectrogram, x_axis='time', y_axis='mel', sr=self.sr, fmax=self.fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()

    @property
    def tensor(self):
        """
        Converts the Mel spectrogram to a PyTorch tensor.
        """
        return torch.tensor(self.mel_spectrogram).unsqueeze(0).float()
