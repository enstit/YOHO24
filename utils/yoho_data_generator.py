#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np

from . import MelSpectrogram


class YOHODataGenerator(DataLoader):
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
        Initializes the data Generator.

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

        print(f"Processing file: {file_path}")
        print(f"Label: {label}")

        # Convert the audio file to a Mel spectrogram
        mel_spectrogram = MelSpectrogram(
            file_path=file_path,
            n_mels=self.n_mels,
            fmax=self.fmax
        )

        print(f"Mel spectrogram shape: {mel_spectrogram.array.shape}")

        # Normalize the Mel spectrogram
        mel_spectrogram.normalize()

        print(
            f"Normalized Mel spectrogram shape: {mel_spectrogram.array.shape}")

        # Convert to tensor: Mel spectrogram and label
        label = torch.tensor(label, dtype=torch.long)

        return mel_spectrogram.tensor, label
