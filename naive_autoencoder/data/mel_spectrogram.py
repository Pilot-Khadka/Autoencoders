import os
import torch
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from noise_generator import NoiseGenerator
from processor import AudioPreprocessor, AudioDataset


SR = 8000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 128


class MelSpectrogramConverter:
    def __init__(
        self, sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.window = torch.hann_window(n_fft, dtype=torch.float32)

        mel_basis_np = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=0.0,
            fmax=sample_rate / 2.0,
            htk=False,
            norm="slaney",
        )
        self.mel_basis = torch.from_numpy(mel_basis_np).float()

    def audio_to_mel(self, audio_waveform):
        """
        :audio_waveform: 1D numpy array or 1D torch tensor (linear amplitude)
        returns: torch.FloatTensor (n_mels, frames) in dB
        """
        if isinstance(audio_waveform, np.ndarray):
            audio_tensor = torch.from_numpy(audio_waveform.astype(np.float32))
        elif isinstance(audio_waveform, torch.Tensor):
            audio_tensor = audio_waveform.to(dtype=torch.float32)
        else:
            raise TypeError("audio_waveform must be numpy array or torch tensor")

        if audio_tensor.ndim != 1:
            audio_tensor = audio_tensor.flatten()

        stft_output = torch.stft(
            input=audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(audio_tensor.device),
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        stft_magnitude = torch.abs(stft_output)

        mel_spec = torch.matmul(
            self.mel_basis.to(stft_magnitude.device), stft_magnitude
        )

        mel_np = mel_spec.cpu().numpy()
        mel_db = librosa.power_to_db(mel_np**2, ref=1.0)

        return torch.from_numpy(mel_db).float()

    def mel_to_audio(self, mel_db, audio_length, n_iter=32):
        """
        Reconstruct audio from Log-Mel (dB).
        :mel_db: numpy array or torch tensor with shape (n_mels, frames) in dB
        :audio_length: desired output sample length (int)

        returns: 1D numpy array (float32) audio waveform
        """
        if isinstance(mel_db, torch.Tensor):
            mel_db = mel_db.cpu().numpy()

        power_spectrogram = librosa.db_to_power(mel_db, ref=1.0)

        S = librosa.feature.inverse.mel_to_stft(
            M=power_spectrogram,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            fmin=0.0,
            fmax=self.sample_rate / 2.0,
            htk=False,
            norm="slaney",
        )

        reconstructed_audio = librosa.griffinlim(
            S=S,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=np.hanning(self.n_fft),
        )

        return reconstructed_audio[:audio_length].astype(np.float32)


def plot_single_spectrogram(
    waveform,
    title="Mel Spectrogram",
    ax=None,
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
):
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    img = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="mel",
        ax=ax,
    )
    ax.set_title(title)

    return img, mel_db


def compare_spectrograms(
    waveforms_dict,
    suptitle="Mel Spectrogram Comparison",
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
):
    n_plots = len(waveforms_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), sharey=True)

    if n_plots == 1:
        axes = [axes]

    for idx, (label, waveform) in enumerate(waveforms_dict.items()):
        img, _ = plot_single_spectrogram(
            waveform,
            title=label,
            ax=axes[idx],
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

    fig.colorbar(img, ax=axes, format="%+2.0f dB")
    fig.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def visualize_noise_comparison(
    clean_waveform,
    noisy_waveforms_dict,
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
):
    all_waveforms = {"Original": clean_waveform}
    all_waveforms.update(noisy_waveforms_dict)

    compare_spectrograms(
        all_waveforms,
        suptitle="Original vs Noisy Audio Comparison",
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )


def get_clean_waveform(dataset, preprocessor, index=0):
    scaled_patch = dataset.data[index]
    return preprocessor.scaler.inverse_transform(scaled_patch.reshape(1, -1))[0].astype(
        np.float32
    )


def generate_noisy_versions(clean_waveform, sample_rate=SR, snr_db=10):
    noise_gen = NoiseGenerator(sample_rate)
    n = len(clean_waveform)

    noisy_versions = {
        "White Noise": clean_waveform + noise_gen.white_noise(n, snr_db),
        "Pink Noise": clean_waveform + noise_gen.pink_noise(n, snr_db),
        "Brown Noise": clean_waveform + noise_gen.brown_noise(n, snr_db),
    }

    return noisy_versions


def show_reconstruction(clean_waveform, sample_rate=SR):
    converter = MelSpectrogramConverter(
        sample_rate=sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    mel_db_tensor = converter.audio_to_mel(clean_waveform)
    mel_np = mel_db_tensor.numpy()

    reconstructed_audio = converter.mel_to_audio(
        mel_np, audio_length=len(clean_waveform), n_iter=32
    )

    compare_spectrograms(
        {"Original": clean_waveform, "Reconstructed": reconstructed_audio},
        suptitle="Original vs Reconstructed Audio",
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    return reconstructed_audio


def run_analysis(
    dataset_dir, audio_index=0, snr_db=10, sample_rate=SR, target_length=SR
):
    preprocessor = AudioPreprocessor(
        target_length=target_length, sample_rate=sample_rate
    )
    rec_dir = os.path.join(
        dataset_dir, "free-spoken-digit-dataset-master", "recordings"
    )
    dataset = AudioDataset(directory=rec_dir, preprocessor=preprocessor)
    scaled_patch = dataset.data[audio_index]
    clean_waveform = preprocessor.scaler.inverse_transform(scaled_patch.reshape(1, -1))[
        0
    ].astype(np.float32)

    plot_single_spectrogram(
        clean_waveform,
        title="Original Audio",
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    plt.show()

    noisy_versions = generate_noisy_versions(clean_waveform, sample_rate, snr_db)

    visualize_noise_comparison(
        clean_waveform,
        noisy_versions,
        sample_rate=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    show_reconstruction(clean_waveform, sample_rate)


def main():
    run_analysis(
        dataset_dir="fsdd",
        audio_index=0,
        snr_db=10,
        sample_rate=SR,
        target_length=SR,
    )


if __name__ == "__main__":
    main()
