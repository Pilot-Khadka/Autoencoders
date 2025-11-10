import os
import torch
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class AudioPreprocessor:
    def __init__(self, target_length=8192, sample_rate=8000):
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.scaler = StandardScaler()

    def load_and_preprocess_audio(self, file_path):
        sr, data = wavfile.read(file_path)

        if data.ndim > 1:
            data = data[:, 0]

        if sr != self.sample_rate:
            factor = self.sample_rate / sr
            new_length = int(len(data) * factor)
            data = np.interp(
                np.linspace(0, len(data), new_length), np.arange(len(data)), data
            )

        if np.max(np.abs(data)) > 0:
            data = data / np.max(np.abs(data))

        return data.astype(np.float32)

    def extract_patches(self, audio_data, patch_length=None, overlap=0.5):
        if patch_length is None:
            patch_length = self.target_length

        audio_len = len(audio_data)

        if audio_len < patch_length:
            pad_amount = patch_length - audio_len
            audio_data = np.pad(audio_data, (0, pad_amount), mode="constant")
            return np.array([audio_data])

        step = max(1, int(patch_length * (1 - overlap)))
        patches = []

        for i in range(0, audio_len - patch_length + 1, step):
            patch = audio_data[i : i + patch_length]
            patches.append(patch)

        if len(patches) == 0:
            patches.append(audio_data[:patch_length])

        return np.array(patches)

    def apply_windowing(self, patches):
        window = np.hanning(patches.shape[1])
        return patches * window[np.newaxis, :]


class AudioDataset(Dataset):
    def __init__(self, directory, preprocessor, noise_std=0.05):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".wav")
        ]
        self.preprocessor = preprocessor
        self.noise_std = noise_std
        self.data = []

        print(f"Loading {len(self.files)} audio files...")

        for file_path in tqdm(self.files, desc="Processing audio files", ncols=80):
            audio = preprocessor.load_and_preprocess_audio(file_path)
            patches = preprocessor.extract_patches(audio)
            patches = preprocessor.apply_windowing(patches)
            self.data.extend(patches)

        if len(self.data) == 0:
            raise ValueError(
                "No patches extracted from any audio files. Check your audio files and target_length."
            )

        self.data = np.array(self.data)
        print(
            f"Dataset loaded: {len(self.data)} patches of length {self.data.shape[1]}"
        )

        preprocessor.scaler.fit(self.data)
        self.data = preprocessor.scaler.transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clean = self.data[idx]
        noise = np.random.normal(0, self.noise_std, clean.shape).astype(np.float32)
        noisy = clean + noise

        return (
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )
