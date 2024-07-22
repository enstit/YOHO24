#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import AudioFile


class YOHODataset(Dataset):
    """
    The YOHODataset class represents a dataset of audio files.
    It provides methods to load the audio files and their labels, and to apply
    transformations to the audio files and labels.
    """

    def __init__(
        self,
        audios: list[AudioFile],
        labels: list[str],
        transform=None,
        target_transform=None,
        n_mels: int = None,
        hop_length: int = None,
        win_length: int = None,
    ):

        self.audios = audios  # List of AudioFile objects representing the audio files
        self.labels = labels  # List of unique labels in the dataset
        # Function to apply to the audio files before returning them
        self.transform = transform
        # Function to apply to the labels before returning them
        self.target_transform = target_transform

        self.n_mels = n_mels  # Number of Mel bins
        self.hop_length = hop_length  # Length of the hop between STFT windows
        self.win_length = win_length  # Length of the STFT window

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):

        # Get the Mel spectrogram of the idx-AudioFile of the dataset
        mel_spectrogram = self.audios[idx].mel_spectrogram(
            n_mels=self.n_mels, hop_length=self.hop_length, win_length=self.win_length
        )

        # Convert the normalized Mel spectrogram to a PyTorch tensor
        normalized_mel_spectrogram_tensor = (
            torch.tensor(mel_spectrogram.normalized).unsqueeze(0).float()
        )

        # Get the labels for the audio file
        labels = self._get_output(idx)

        return normalized_mel_spectrogram_tensor, labels

    def _get_output(self, idx: int) -> np.array:

        STEP_SIZE = 0.3125

        duration = self.audios[idx].duration

        output_size = ((len(self.labels) * 3), int(duration // STEP_SIZE))

        output = np.empty(output_size)

        # Initialize columns equal to 1 module 3 to 0
        output[1::3] = 0

        timeadvancement_no = 0
        while timeadvancement_no < output.shape[1]:
            window_start = timeadvancement_no * STEP_SIZE
            window_end = (timeadvancement_no + 1) * STEP_SIZE

            for audio_label in self.audios[idx].labels:
                if (audio_label[1] <= window_start <= audio_label[2]) or (
                    audio_label[1] <= window_end <= audio_label[2]
                ):
                    normalized_start = max(0, audio_label[1] - window_start) / STEP_SIZE
                    normalized_end = (
                        min(STEP_SIZE, audio_label[2] - window_start) / STEP_SIZE
                    )

                    label_index = self.labels.index(audio_label[0])
                    output[label_index * 3, timeadvancement_no] = 1
                    output[label_index * 3 + 1, timeadvancement_no] = normalized_start
                    output[label_index * 3 + 2, timeadvancement_no] = normalized_end

            timeadvancement_no += 1

        return output


class TUTDataset(YOHODataset):

    def __init__(
        self,
        audios: list[AudioFile],
        transform=None,
        target_transform=None,
    ):
        # The TUTYOHODataset class is a subclass of the YOHODataset class
        # where the number of Mel bins is set to 40, the hop length is set to 441,
        # and the window length is set to 1764 as specified in the original
        # YOHO paper. The labels are the ones from the TUT challenge.
        super().__init__(
            audios=audios,
            labels=[
                "brakes squeaking",
                "car",
                "children",
                "large vehicle",
                "people speaking",
                "people walking",
            ],
            transform=transform,
            target_transform=target_transform,
            n_mels=40,
            hop_length=441,
            win_length=1764,
        )


class YOHODataGenerator(DataLoader):
    def __init__(
        self,
        dataset: YOHODataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
