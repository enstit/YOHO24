#!/usr/bin/env python

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from . import AudioFile


class YOHODataset(Dataset):
    def __init__(self, audios: list[AudioFile], transform=None, target_transform=None):
        self.audios = audios
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):

        audio = self.audios[idx]

        mel_spectrogram = audio.mel_spectrogram
        labels = eval(audio.labels)

        # Normalize the Mel spectrogram
        if not mel_spectrogram.is_normalized:
            mel_spectrogram.normalize()

        mel_spectrogram = mel_spectrogram.tensor

        # if self.transform:
        #    mel_spectrogram = self.transform(mel_spectrogram)
        # if self.target_transform:
        #    label = self.target_transform(label)

        return mel_spectrogram, labels


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
