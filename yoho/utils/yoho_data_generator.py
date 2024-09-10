import numpy as np
import torch
import pickle
import os
from torch.utils.data import Dataset, DataLoader

from yoho.utils import AudioFile

SCRIPT_DIRPATH = os.path.abspath(os.path.dirname(__file__))


class YOHODataset(Dataset):
    """
    The YOHODataset class is used to construct audio Dataset for the YOHO model.
    """

    def __init__(
        self,
        audios: list[AudioFile],
        labels: list[str],
        transform=None,
        target_transform=None,
        n_mels: int = None,
        hop_len: float = None,
        win_len: float = None,
    ):

        # Check that all the AudioFiles have the same sample rate and duration
        sample_rate = audios[0].sr
        duration = audios[0].duration
        for audioclip in audios:
            if not (
                audioclip.sr == sample_rate or audioclip.duration == duration
            ):
                raise ValueError(
                    "All AudioFiles must have the same duration and sample rate"
                )

        self.audios = (
            audios  # List of Audios objects representing the audio files
        )
        self.labels = labels  # List of unique labels in the dataset
        # Function to apply to the audio files before returning them
        self.transform = transform
        # Function to apply to the labels before returning them
        self.target_transform = target_transform
        self.hop_len = hop_len  # Hop length in seconds
        self.win_len = win_len  # Window length in seconds
        self.n_mels = n_mels  # Number of Mel bins

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):

        # Get the Mel spectrogram of the idx-AudioClip of the dataset
        spect = self.audios[idx].mel_spectrogram(
            n_mels=self.n_mels,
            hop_len=self.hop_len,
            win_len=self.win_len,
            normalized=True,
        )

        # Convert the normalized Mel spectrogram to a PyTorch tensor
        spect_tensor = torch.tensor(spect).unsqueeze(0).float()

        # Get the labels for the audio file
        labels = self._get_output(idx)

        if self.transform:
            spect_tensor = self.transform(spect_tensor)

        return spect_tensor, labels

    def _get_output(self, idx: int) -> np.array:

        STEPS_NO = 9  # Number of steps in the output

        step_duration = self.audios[idx].duration / STEPS_NO

        output_size = (3 * len(self.labels), STEPS_NO)

        output = np.zeros(output_size)

        # Initialize class columns to 0
        output[0::3, :] = 0

        timeadvancement_no = 0
        while timeadvancement_no < output.shape[1]:
            window_start = timeadvancement_no * step_duration
            window_end = (timeadvancement_no + 1) * step_duration

            for audio_label in (
                self.audios[idx].labels if self.audios[idx].labels else []
            ):
                if (audio_label[1] <= window_start <= audio_label[2]) or (
                    audio_label[1] <= window_end <= audio_label[2]
                ):
                    normalized_start = (
                        max(0, audio_label[1] - window_start) / step_duration
                    )
                    normalized_end = (
                        min(step_duration, audio_label[2] - window_start)
                        / step_duration
                    )

                    label_index = self.labels.index(audio_label[0])
                    output[label_index * 3, timeadvancement_no] = 1
                    output[label_index * 3 + 1, timeadvancement_no] = (
                        normalized_start
                    )
                    output[label_index * 3 + 2, timeadvancement_no] = (
                        normalized_end
                    )

            timeadvancement_no += 1

        return output

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def load(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")

        with open(filepath, "rb") as f:
            return pickle.load(f)


class UrbanSEDDataset(YOHODataset):

    def __init__(
        self,
        audios: list[AudioFile],
        transform=None,
        target_transform=None,
    ):
        # The UrbanSEDDataset class is a subclass of the YOHODataset class
        # where the number of Mel bins is set to 40, the hop length is set to
        # 10 ms, and the window length is set to 40 ms as specified in the
        # original YOHO paper. The labels are the ones from the URBAN-SED
        # challenge.
        super().__init__(
            audios=audios,
            labels=[
                "noise",
                "air_conditioner",
                "car_horn",
                "children_playing",
                "dog_bark",
                "drilling",
                "engine_idling",
                "gun_shot",
                "jackhammer",
                "siren",
                "street_music",
            ],
            transform=transform,
            target_transform=target_transform,
            n_mels=40,  # As defined in the YOHO paper for the URBAN-SED dataset
            hop_len=0.01,  # As defined in the YOHO paper for the URBAN-SED dataset
            win_len=0.04,  # As defined in the YOHO paper for the URBAN-SED dataset
        )


class TUTDataset(YOHODataset):

    def __init__(
        self,
        audios: list[AudioFile],
        transform=None,
        target_transform=None,
    ):
        # The TUTDataset class is a subclass of the YOHODataset class
        # where the number of Mel bins is set to 40, the hop length is set to
        # 10 ms, and the window length is set to 40 ms as specified in the
        # original YOHO paper. The labels are the ones from the TUT challenge.
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
            n_mels=40,  # As defined in the YOHO paper for the TUT dataset
            hop_len=0.01,  # As defined in the YOHO paper for the TUT dataset
            win_len=0.04,  # As defined in the YOHO paper for the TUT dataset
        )


class YOHODataGenerator(DataLoader):
    def __init__(
        self,
        dataset: YOHODataset,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = False
    ):
        super().__init__(
            dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory
        )
        self.n_classes = len(
            dataset.labels
        )  # Number of classes in the dataset
