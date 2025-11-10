import os
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt

from download_dataset import download_and_extract_dataset


class NoiseGenerator:
    def __init__(self, sample_rate: int = 8000):
        self.sr = sample_rate

    def white_noise(self, n: int, snr_db):
        """random sampling from normal distribution"""

        # flat spectral density
        # equal energy at all frequencies
        noise = np.random.normal(0, 1, n)
        return self._adjust_snr(noise, snr_db)

    def pink_noise(self, n, snr_db: float):
        """1/f noise
        high frequency bins get smaller amplitude
        low frequenc bins remain stronger"""
        white = np.random.randn(n)

        # gives frequency bins of DFT
        freqs = np.fft.fftfreq(n, 1 / self.sr)
        freqs[0] = 1  # avoid division by 0

        # representation of freq. of white noise
        fft_white = np.fft.fft(white)
        fft_pink = fft_white / np.sqrt(np.abs(freqs))
        pink = np.real(np.fft.ifft(fft_pink))
        return self._adjust_snr(pink, snr_db)

    def brown_noise(self, n, snr_db: float):
        white = np.random.randn(n)
        freqs = np.fft.fftfreq(n, 1 / self.sr)
        freqs[0] = 1
        fft_white = np.fft.fft(white)
        fft_brown = fft_white / np.abs(freqs)
        brown = np.real(np.fft.ifft(fft_brown))
        return self._adjust_snr(brown, snr_db)

    def impulsive_noise(self, signal_length, snr_db: float, impulse_prob: float = 0.01):
        """simulates clicks, pops, or digital artifacts"""
        base_noise = np.random.normal(0, 0.1, signal_length)

        impulse_mask = np.random.random(signal_length) < impulse_prob
        impulses = np.random.normal(0, 5, signal_length) * impulse_mask

        noise = base_noise + impulses
        return self._adjust_snr(noise, snr_db)

    def colored_noise_mix(self, signal_length, snr_db: float):
        white = self.white_noise(signal_length, snr_db + 6)
        pink = self.pink_noise(signal_length, snr_db + 6)
        brown = self.brown_noise(signal_length, snr_db + 6)

        w1, w2, w3 = np.random.dirichlet([1, 1, 1])
        mixed = w1 * white + w2 * pink + w3 * brown

        return self._adjust_snr(mixed, snr_db)

    def _adjust_snr(self, noise, target_snr_db: float):
        """
        SNR = 10. log10(Psignal/Pnoise)
        """
        noise_power = np.mean(noise**2)

        if noise_power == 0:
            return noise

        target_power = 10 ** (-target_snr_db / 10)
        scale = np.sqrt(target_power / noise_power)
        return noise * scale


def load_fsdd_dataset(dataset_dir, max_files=None):
    rec_dir = os.path.join(
        dataset_dir, "free-spoken-digit-dataset-master", "recordings"
    )
    files, labels = [], []
    for f in os.listdir(rec_dir):
        if f.endswith(".wav"):
            path = os.path.join(rec_dir, f)
            files.append(path)
            labels.append(int(f.split("_")[0]))
            if max_files and len(files) >= max_files:
                break
    return files, labels


def play_audio(data, sr):
    data = data.astype(np.float32) / np.max(np.abs(data))
    sd.play(data, sr)
    sd.wait()


def visualize_noises(noises, sr):
    plt.figure(figsize=(12, 8))

    for i, (name, noise) in enumerate(noises.items(), start=1):
        plt.subplot(3, 2, 2 * i - 1)
        plt.plot(noise)
        plt.title(f"{name.capitalize()} Noise (Time Domain)")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        freqs = np.fft.rfftfreq(len(noise), 1 / sr)
        psd = np.abs(np.fft.rfft(noise)) ** 2
        plt.subplot(3, 2, 2 * i)
        plt.semilogy(freqs, psd)
        plt.title(f"{name.capitalize()} Noise (Frequency Domain)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power")

    plt.tight_layout()
    plt.show()


def test_noise_on_sample(audio_path: str, snr_db: float = 10, visualize: bool = True):
    sr, data = wavfile.read(audio_path)
    data = data.astype(np.float32) / np.max(np.abs(data))
    n = len(data)
    gen = NoiseGenerator(sr)

    noises = {
        "white": gen.white_noise(n, snr_db),
        "pink": gen.pink_noise(n, snr_db),
        "brown": gen.brown_noise(n, snr_db),
    }

    if visualize:
        visualize_noises(noises, sr)

    play_audio(data, sr)

    for name, noise in noises.items():
        play_audio(data + noise, sr)


def main():
    dataset_dir = "fsdd"
    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    rec_dir = os.path.join(
        dataset_dir, "free-spoken-digit-dataset-master", "recordings"
    )

    if not os.path.exists(rec_dir):
        download_and_extract_dataset(url, dataset_dir)

    audio_files, _ = load_fsdd_dataset(dataset_dir)
    test_noise_on_sample(audio_files[0])


if __name__ == "__main__":
    main()
