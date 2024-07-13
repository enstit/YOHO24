#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
import numpy as np

from . import AudioFile


class YOHODataGenerator(DataLoader):
    def __init__(
        self,
        file_paths: list,
        labels: list[tuple],
        input_shape: tuple,
        output_shape: tuple,
        batch_size: int = 32,
        shuffle: bool = True,

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

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):

        audio_file = AudioFile(
            file_path=self.file_paths[idx], labels=self.labels[idx])

        # Normalize the Mel spectrogram
        if not audio_file.mel_spectrogram.is_normalized:
            audio_file.mel_spectrogram.normalize()

        return audio_file.mel_spectrogram.tensor, audio_file.labels
