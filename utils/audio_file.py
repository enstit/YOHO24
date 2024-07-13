import librosa
import numpy as np
from matplotlib import pyplot as plt


class AudioFile:
    """
    The AudioFile class represents an audio file.
    It provides methods to load the audio file, extract features, and plot the
    waveform and the Mel spectrogram.
    """

    def __init__(self, filepath: str, labels: list = None, duration: float = None, n_channels: int = 1, sr: int = 44_100):
        """
        Initializes the AudioFile class.

        Args:
            filepath (str): Path to the audio file.
            labels (list): List of labels for the audio file.
            duration (float): Duration of the audio file, in seconds.
            n_channels (int): Number of channels in the audio file.
            sr (int): Sampling rate for the audio file.
        """
        self.filepath = filepath  # Path to the audio file
        self.labels = labels  # List of labels for the audio file
        self.duration = duration  # Duration of the audio file, in seconds
        self.n_channels = n_channels  # Number of channels in the audio file
        self.sr = sr  # Sampling rate for the audio file

    @property
    def frequency_bins(self):
        raise NotImplementedError

    @property
    def waveform(self):
        y, _ = librosa.load(self.filepath, sr=self.sr)
        return y

    @property
    def mel_spectrogram(self):
        return MelSpectrogram(audiofile=self)

    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.waveform)
        plt.title(f'Audio waveform: {self.filepath}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (dB)')
        # Format the x labels to show the seconds in the [MM:SS.sss] format (until the last tick that shows the total duration)
        plt.xticks(
            np.arange(0, len(self.waveform), step=int(
                len(self.waveform) / 10)),
            [f'{int(i / self.sr // 60):02d}:{int(i / self.sr % 60):02d}.{int(i % self.sr):03d}' for i in np.arange(
                0, len(self.waveform), step=int(len(self.waveform) / 10))],
            rotation=45, ha='right'
        )
        plt.tight_layout()
        plt.show()


class MelSpectrogram:
    """
    The MelSpectrogram class represents a Mel spectrogram.
    It provides methods to compute and plot the Mel spectrogram of a given
    audio file.
    """

    def __init__(self, audiofile: AudioFile, n_mels: int = 64, hop_lenght: int = 10, win_length: int = 25, sr: int = 44_100, fmin: int = 0, fmax: int = 7_500):
        """
        Initializes the MelSpectrogram class.

        Args:
            n_mels (int): The number of Mel bands to generate.
            hop_lenght (int): Number of samples between successive frames (hop size).
            fmin (int): Minimum frequency in Hz.
            fmax (int): Maximum frequency in Hz.
            sr (int): Sampling rate for the audio file.
            hop_length (int): Number of samples between successive frames (hop size).
            win_length (int): Size of the FFT window (window size).
        """
        self.audiofile = audiofile
        self.n_mels = n_mels
        self.hop_lenght = hop_lenght
        self.win_length = win_length
        self.sr = sr
        self.fmin = fmin
        self.fmax = fmax
        self.raw = None
        self._compute_spectrogram()

    @property
    def normalized(self) -> np.ndarray:
        """
        Return the normalized Mel spectrogram.
        """
        return (self.raw - np.mean(self.raw)) / np.std(self.raw)

    def _compute_spectrogram(self, mono: bool = True) -> np.ndarray:
        """
        Converts a wav file to a Mel spectrogram.

        Input:
        - mono (bool): Whether to convert the audio to mono.

        Returns:
        - np.ndarray: The Mel spectrogram.
        """

        # Load the audio file using the native sampling rate
        y, origin_sr = librosa.load(self.audiofile.filepath, sr=None)

        # Resample the target sample rate
        if origin_sr != self.sr:
            y = librosa.resample(y=y, orig_sr=origin_sr, target_sr=self.sr)

        # Convert to mono by averaging channels (if needed)
        if mono is True and y.ndim > 1:
            y = np.mean(y, axis=0)

        # Compute the Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_length=self.hop_lenght,
            win_length=self.win_length
        )

        # Convert the Mel spectrogram to a log scale (dB)
        mel_spectrogram = librosa.power_to_db(
            mel_spectrogram, ref=np.max)

        self.raw = mel_spectrogram

    def plot(self):
        """
        Plots the Mel spectrogram.
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            self.mel_spectrogram, x_axis='time', y_axis='mel', sr=self.sr, fmax=self.fmax)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()
