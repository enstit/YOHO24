import numpy as np
from matplotlib import pyplot as plt
import librosa
from IPython.display import Audio


class AudioFile:
    """
    The AudioFile class represents an audio file on the filesystem.
    """

    def __init__(
        self,
        filepath: str,
        labels: list[tuple[str, float, float]] = None,
    ):
        """
        Initializes the AudioFile class.

        Args:
            filepath (str): Path to the audio file on the filesystem
            labels (list): List of labels for the audio file with related start
            and stop times
            n_channels (int): Number of channels in the audio file
        """
        self.filepath = filepath  # Path to the audio file
        self.labels = labels  # List of labels for the audio file with related start and stop times

    def __repr__(self):
        return f"{self.__class__.__name__}(filepath={self.filepath})"

    @property
    def duration(self):
        return librosa.get_duration(path=self.filepath)

    @property
    def sr(self):
        return librosa.get_samplerate(path=self.filepath)

    @property
    def waveform(self):
        y, _ = librosa.load(self.filepath, sr=self.sr, mono=True)
        return y

    def play(self):
        return Audio(self.waveform, rate=self.sr, autoplay=True)

    def mel_spectrogram(
        self,
        n_mels: int = 64,
        win_len: float = 1.00,
        hop_len: float = 1.00,
        normalized: bool = True,
    ):
        spect = librosa.feature.melspectrogram(
            y=self.waveform,
            sr=self.sr,
            n_mels=n_mels,
            win_length=int(win_len * self.sr),
            hop_length=int(hop_len * self.sr),
        )

        # Convert the Mel spectrogram to a log scale (dB)
        spect = librosa.power_to_db(spect, ref=np.max)

        if normalized:
            spect = (spect - np.mean(spect)) / np.std(spect)

        return spect

    def subdivide(self, win_len: float = 1.00, hop_len: float = 1.00):
        """
        Subdivides the audio file into AudioClips of a given window size and hop size.

        Args:
            win_len (float, optional): number of seconds of the window. Defaults to 1.00.
            hop_len (float, optional): number of seconds of the hop. Defaults to 1.00.

        Returns:
            list: list of the AudioClips obtained by subdividing the audio file.
        """

        if win_len <= 0 or hop_len <= 0:
            raise ValueError("Both win_ms and hop_ms must be positive")

        win_points = int(win_len * self.sr)
        hop_points = int(hop_len * self.sr)

        audioclips = []

        for i in range(
            win_points, int(self.duration * self.sr) + 1, hop_points
        ):

            labels = []
            for label in self.labels if self.labels else []:
                # Check if the label is within the current window.
                # If so, add it to the labels list by changing the start and stop times
                # to be relative to the current window.
                if (
                    label[1] < i / self.sr
                    and label[2] > (i - win_points) / self.sr
                ):
                    labels.append(
                        (
                            label[0],
                            max(0, label[1] - (i - win_points) / self.sr),
                            min(
                                win_points / self.sr,
                                label[2] - (i - win_points) / self.sr,
                            ),
                        )
                    )

            audioclips.append(
                AudioClip(
                    filepath=self.filepath,
                    labels=labels,
                    offset=(i - win_points) / self.sr,
                    duration=win_points / self.sr,
                )
            )

        return audioclips

    def plot_labels(self):
        # TODO: Refactor this method
        plt.figure(figsize=(20, 6))

        librosa.display.specshow(
            data=self.mel_spectrogram(n_mels=40, win_len=0.04, hop_len=0.01),
            sr=self.sr,
            x_axis="frames",
            y_axis="mel",
        )

        for label in self.labels:
            # Calculate the initial frame and the final frame of the label
            # with the same window size and hop size used to calculate the Mel spectrogram
            # (win_len=0.04, hop_len=0.01)
            frames_n = self.mel_spectrogram(
                n_mels=40, win_len=0.04, hop_len=0.01
            ).shape[1]
            WINDOW_SIZE = 2.56
            normalized_start = int(label[1] * frames_n / WINDOW_SIZE)
            normalized_stop = int(label[2] * frames_n / WINDOW_SIZE) - 1
            plt.axvspan(
                normalized_start,
                normalized_stop,
                facecolor="red",
                alpha=0.25,
                label=label[0],
            )
            plt.text(
                x=normalized_start + 2,
                y=44_100 / 16,
                s=label[0],
                fontsize=12,
                color="white",
                ha="center",
                va="center",
                rotation=90,
            )

        plt.title("Audio waveform with labels")
        plt.xlabel("Time (frames)")
        plt.ylabel("Amplitude")
        plt.show()


class AudioClip(AudioFile):

    def __init__(
        self,
        filepath: str,
        offset: float,
        duration: float,
        labels: list = None,
    ):
        super().__init__(filepath=filepath, labels=labels)
        self.offset = offset
        self.duration_ = duration

    @property
    def duration(self):
        return self.duration_

    @property
    def waveform(self):

        y, _ = librosa.load(
            self.filepath,
            sr=self.sr,
            mono=True,
            offset=max(0, self.offset),
            duration=self.duration,
        )

        if self.offset < 0:
            y = np.concatenate(
                [
                    np.zeros((int(-self.offset * self.sr),)),
                    y,
                ]
            )

        if self.duration_ > y.shape[0] / self.sr:
            y = np.concatenate(
                [
                    y,
                    np.zeros((int(self.duration * self.sr - y.shape[0]),)),
                ]
            )

        return y


def plot_melspectrogram(mel, sr):
    """
    Plots the Mel spectrogram.
    """
    plt.figure(figsize=(10, 4))
    plt.title(f"Mel spectrogram")
    librosa.display.specshow(data=mel, sr=sr, x_axis="frames", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()
