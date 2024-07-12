
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.mel_spectrogram import wav_to_mel_spectrogram


class AudioDataset(DataLoader):
    def __init__(
        self,
        file_paths: list,
        labels: list[tuple],
        input_shape: tuple,
        output_shape: tuple,
        batch_size: int = 32,
        n_mels: int = 64,
        fmax: int = 7500
    ):
        """
        Initializes the AudioDataset.

        Args:
            file_paths (list): List of file paths to the audio files.
            labels (list): List of corresponding labels.
            input_shape (tuple): Shape of the input tensor (channels, height, width).
            output_shape (tuple): Shape of the output tensor (channels, height, width).
            n_mels (int): Number of Mel bands to generate.
            fmax (int): Maximum frequency in Hz.
          """

        self.file_paths = file_paths
        self.labels = labels
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.n_mels = n_mels
        self.fmax = fmax

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Convert the audio file to a Mel spectrogram
        mel_spectrogram, _ = wav_to_mel_spectrogram(
            file_path=file_path,
            n_mels=self.n_mels,
            fmax=self.fmax
        )

        # Normalize the Mel spectrogram
        mel_spectrogram = (mel_spectrogram -
                           np.mean(mel_spectrogram)) / np.std(mel_spectrogram)

        # Ensure the Mel spectrogram has the right shape
        if mel_spec.shape[1] < self.input_shape[2]:
            """
            If the number of time steps in the Mel spectrogram (mel_spec.shape[1]) is less than the required width 
            (self.input_shape[2]), the spectrogram is padded with zeros (constant padding) along the time dimension to match the required width.
            """
            mel_spec = np.pad(
                mel_spec, ((0, 0), (0, self.input_shape[2] - mel_spec.shape[1])), mode='constant')
        elif mel_spec.shape[1] > self.input_shape[2]:
            """
            If the number of time steps in the Mel spectrogram is greater than the required width, 
            the spectrogram is truncated by slicing it to match the required width.
            """
            mel_spec = mel_spec[:, :self.input_shape[2]]

        # Convert to tensor: Mel spectrogram and label
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return mel_spectrogram, label
