import librosa
import numpy as np
from matplotlib import pyplot as plt


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
        n_channels: int = 1,
    ):
        """
        Initializes the AudioFile class.

        Args:
            filepath (str): Path to the audio file
            labels (list): List of labels for the audio file with related start and stop times
            n_channels (int): Number of channels in the audio file
        """
        self.filepath = filepath  # Path to the audio file
        self.labels = labels  # List of labels for the audio file with related start and stop times
        self.n_channels = n_channels  # Number of channels in the audio file

    @property
    def duration(self):
        return librosa.get_duration(path=self.filepath)

    @property
    def sr(self):
        return librosa.get_samplerate(path=self.filepath)

    @property
    def waveform(self):
        y, _ = librosa.load(self.filepath, sr=self.sr)
        return y

    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.waveform)
        plt.title(f'Audio waveform: {self.filepath}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (dB)')
        # Format the x labels to show the seconds in the [MM:SS.sss] format (until the last tick that shows the total duration)
        plt.xticks(
            np.arange(0, len(self.waveform) + 1, step=int(
                len(self.waveform) / 10)),
            [f"[{int(i / self.sr // 60):02d}:{int(i / self.sr % 60):02d}.{int(i % self.sr):03d}]" for i in np.arange(
                0, len(self.waveform) + 1, step=int(len(self.waveform) / 10))],
            rotation=60, ha='right'
        )
        plt.tight_layout()
        plt.show()

    def mel_spectrogram(self, n_mels: int = 64, hop_length: int = 10, win_length: int = 25):
        return MelSpectrogram(audiofile=self, n_mels=n_mels, hop_length=hop_length, win_length=win_length)


class MelSpectrogram:
    """
    The MelSpectrogram class represents a Mel spectrogram.
    It provides methods to compute and plot the Mel spectrogram of a given
    audio file.
    """

    def __init__(self, audiofile: AudioFile, n_mels: int = 64, hop_length: int = 10, win_length: int = 25):
        """
        Initializes the MelSpectrogram class.

        Args:
            n_mels (int): The number of Mel bands to generate.
            hop_length (int): Number of samples between successive frames (hop size).
            win_length (int): Size of the FFT window (window size).
        """
        self.audiofile = audiofile
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
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
            y=self.audiofile.waveform,
            sr=self.audiofile.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length
        )

        # Convert the Mel spectrogram to a log scale (dB)
        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram, ref=np.max)

        self.raw = mel_spectrogram

    @property
    def shape(self):
        return self.raw.shape

    def plot(self):
        """
        Plots the Mel spectrogram.
        """
        plt.figure(figsize=(10, 4))
        plt.title(f'Mel spectrogram: {self.audiofile.filepath}')
        librosa.display.specshow(
            data=self.raw, sr=self.audiofile.sr, x_axis='frames', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()
