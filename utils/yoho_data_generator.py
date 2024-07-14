#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader

from . import AudioFile


class YOHODataset(Dataset):
    """
    The YOHODataset class represents a dataset of audio files.
    It provides methods to load the audio files and their labels, and to apply
    transformations to the audio files and labels.
    """

    def __init__(self, audios: list[AudioFile], labels: list[str], transform=None, target_transform=None, n_mels: int = None, hop_length: int = None, win_length: int = None):

        self.audios = audios  # List of AudioFile objects representing the audio files
        self.labels = labels  # List of unique labels in the dataset
        # Function to apply to the audio files before returning them
        self.transform = transform
        # Function to apply to the labels before returning them
        self.target_transform = target_transform

        self.n_mels = n_mels  # Number of Mel bins
        self.hop_length = hop_length
        self.win_length = win_length

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):

        # Get the Mel spectrogram of the idx-AudioFile of the dataset
        mel_spectrogram = self.audios[idx].mel_spectrogram(
            n_mels=self.n_mels, hop_length=self.hop_length, win_length=self.win_length)

        # Convert the normalized Mel spectrogram to a PyTorch tensor
        normalized_mel_spectrogram_tensor = torch.tensor(
            mel_spectrogram.normalized).unsqueeze(0).float()

        # Get the labels for the audio file
        labels = self.audios[idx].labels

        return normalized_mel_spectrogram_tensor, labels


class YOHODataGenerator(DataLoader):
    def __init__(
        self,
        dataset: YOHODataset,
        batch_size: int = 32,
        shuffle: bool = True,

    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
