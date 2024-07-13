#!/usr/bin/env python

import pandas as pd
from torch.utils.data import Dataset, DataLoader

from . import AudioFile


class YOHODataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        audio_file = AudioFile(
            file_path=self.annotations.filepath[idx], labels=self.annotations.events[idx])

        mel_spectrogram = audio_file.mel_spectrogram
        labels = eval(audio_file.labels)

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
