#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import AudioClip, MelSpectrogram


class YOHODataset(Dataset):
    """
    The YOHODataset class represents a dataset of audio files.
    It provides methods to load the audio files and their labels, and to apply
    transformations to the audio files and labels.
    """

    def __init__(
        self,
        audioclips: list[AudioClip],
        labels: list[str],
        transform=None,
        target_transform=None,
        n_mels: int = None,
        hop_ms: float = None,
        win_ms: float = None,
    ):

        self.audioclips = (
            audioclips  # List of AudioClip objects representing the audio files
        )
        self.labels = labels  # List of unique labels in the dataset
        # Function to apply to the audio files before returning them
        self.transform = transform
        # Function to apply to the labels before returning them
        self.target_transform = target_transform
        self.hop_ms = hop_ms  # Hop length in milliseconds
        self.win_ms = win_ms  # Window length in milliseconds

        self.n_mels = n_mels  # Number of Mel bins

    def __len__(self):
        return len(self.audioclips)

    def __getitem__(self, idx):

        # Get the Mel spectrogram of the idx-AudioClip of the dataset
        mel_spectrogram = MelSpectrogram(
            self.audioclips[idx].waveform,
            n_mels=self.n_mels,
            hop_ms=self.hop_ms,
            win_ms=self.win_ms,
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

        duration = self.audioclips[idx].duration

        output_size = ((len(self.labels) * 3), int(duration // STEP_SIZE))

        output = np.empty(output_size)

        # Initialize columns equal to 1 module 3 to 0
        output[1::3] = 0

        timeadvancement_no = 0
        while timeadvancement_no < output.shape[1]:
            window_start = timeadvancement_no * STEP_SIZE
            window_end = (timeadvancement_no + 1) * STEP_SIZE

            for audio_label in (
                self.audioclips[idx].labels if self.audioclips[idx].labels else []
            ):
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
        audioclips: list[AudioClip],
        transform=None,
        target_transform=None,
    ):
        # The TUTYOHODataset class is a subclass of the YOHODataset class
        # where the number of Mel bins is set to 40, the hop length is set to 441,
        # and the window length is set to 1764 as specified in the original
        # YOHO paper. The labels are the ones from the TUT challenge.
        super().__init__(
            audioclips=audioclips,
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
            n_mels=40,  # As defined in the YOHO paper for the TUT dataset
            hop_ms=10,  # As defined in the YOHO paper for the TUT dataset
            win_ms=40,  # As defined in the YOHO paper for the TUT dataset
        )


class YOHODataGenerator(DataLoader):
    def __init__(
        self,
        dataset: YOHODataset,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
