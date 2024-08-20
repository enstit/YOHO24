import librosa
import math
import numpy as np
from matplotlib import pyplot as plt


class AudioClip:

    def __init__(self, waveform: list[float], sr: int, labels: list = None):
        self.waveform = waveform
        self.sr = sr
        self.labels = labels

    @property
    def duration(self):
        """
        Returns the duration of the audio clip in seconds.
        """
        return self.waveform.shape[0] / self.sr


class AudioFile:
    """
    The AudioFile class represents an audio file.
    It provides methods to load the audio file, extract features, and plot the
    waveform and the Mel spectrogram.
    """

    def __init__(
        self,
        filepath: str,
        labels: list = None,
    ):
        """
        Initializes the AudioFile class.

        Args:
            filepath (str): Path to the audio file
            labels (list): List of labels for the audio file with related start and stop times
            n_channels (int): Number of channels in the audio file
        """
        self.filepath = filepath  # Path to the audio file
        self.labels = eval(
            labels
        )  # List of labels for the audio file with related start and stop times

    @property
    def duration(self):
        return librosa.get_duration(path=self.filepath)

    @property
    def sr(self):
        return librosa.get_samplerate(path=self.filepath)

    @property
    def waveform(self):
        y, _ = librosa.load(self.filepath, sr=self.sr)
        return librosa.to_mono(y)  # Convert the audio to mono

    def audioclips(self, win_ms: float = None, hop_ms: float = None):
        """

        Args:
            win_len (float, optional): number of seconds of the window. Defaults to 2.56.
            hop_len (float, optional): number of seconds of the hop. Defaults to 1.00.

        Returns:
            list: list of the audio samples
            list: list of the time ranges of each window
        """

        if win_ms <= 0 or hop_ms <= 0:
            raise ValueError("Both win_ms and hop_ms must be positive")

        win_points = int(win_ms / 1000 * self.sr)
        hop_points = int(hop_ms / 1000 * self.sr)

        if self.waveform.shape[0] < win_points:
            waveform_padded = np.zeros((win_points,))
            waveform_padded[: self.waveform.shape[0]] = self.waveform

        else:
            hops_no = math.ceil((self.waveform.shape[0] - win_points) / hop_points)
            waveform_padded = np.zeros((int(win_points + hops_no * hop_points),))
            waveform_padded[: self.waveform.shape[0]] = self.waveform

        windows_waveforms = [
            AudioClip(waveform_padded[i - win_points : i], self.sr)
            for i in range(win_points, waveform_padded.shape[0] + 1, hop_points)
        ]
        windows_ranges = [
            ((i - win_points) / self.sr, i / self.sr)
            for i in range(win_points, waveform_padded.shape[0] + 1, hop_points)
        ]

        return windows_waveforms, windows_ranges

    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.waveform)
        plt.title(f"Audio waveform: {self.filepath}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (dB)")
        # Format the x labels to show the seconds in the [MM:SS.sss] format (until the last tick that shows the total duration)
        plt.xticks(
            np.arange(0, len(self.waveform) + 1, step=int(len(self.waveform) / 10)),
            [
                f"[{int(i / self.sr // 60):02d}:{int(i / self.sr % 60):02d}.{int(i % self.sr):03d}]"
                for i in np.arange(
                    0, len(self.waveform) + 1, step=int(len(self.waveform) / 10)
                )
            ],
            rotation=60,
            ha="right",
        )
        plt.tight_layout()
        plt.show()


class MelSpectrogram:
    """
    The MelSpectrogram class represents a Mel spectrogram.
    It provides methods to compute and plot the Mel spectrogram of a given
    audio file.
    """

    def __init__(
        self,
        waveform: list[float],  # The audio waveform as a 1D numpy array
        sr: int = 44_100,  # The sample rate of the audio file
        n_mels: int = 64,  # Number of Mel bands to generate
        hop_ms: int = None,  # Number of milliseconds between successive frames (hop size).
        win_ms: int = None,  # Size, in milliseconds, of the FFT window (window size).
    ):
        self.waveform = waveform
        self.sr = sr
        self.n_mels = n_mels
        self.hop_ms = hop_ms
        self.win_ms = win_ms
        self.raw = None
        self._compute_spectrogram()

    @property
    def normalized(self) -> np.ndarray:
        """
        Return the normalized Mel spectrogram.
        """
        return (self.raw - np.mean(self.raw)) / np.std(self.raw)

    def _compute_spectrogram(self) -> np.ndarray:
        """
        Converts a wav file to a Mel spectrogram.

        Input:
        - mono (bool): Whether to convert the audio to mono.

        Returns:
        - np.ndarray: The Mel spectrogram.
        """

        # Compute the Mel spectrogram with provided parameters
        mel_spectrogram = librosa.feature.melspectrogram(
            y=self.waveform,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_ms * self.sr // 1000,
            win_length=self.win_ms * self.sr // 1000,
        )

        # Convert the Mel spectrogram to a log scale (dB)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        self.raw = mel_spectrogram

    @property
    def shape(self):
        return self.raw.shape

    def plot(self):
        """
        Plots the Mel spectrogram.
        """
        plt.figure(figsize=(10, 4))
        plt.title(f"Mel spectrogram")
        librosa.display.specshow(
            data=self.raw, sr=self.sr, x_axis="frames", y_axis="mel"
        )
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
