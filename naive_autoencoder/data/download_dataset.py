import os
import zipfile
import requests
import numpy as np
from tqdm import tqdm
import sounddevice as sd
from scipy.io import wavfile
from typing import Union


def download_and_extract_dataset(url: str, dataset_dir: str):
    zip_path = os.path.join(dataset_dir, "temp.zip")
    os.makedirs(dataset_dir, exist_ok=True)

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with (
        open(zip_path, "wb") as f,
        tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading FSDD"
        ) as pbar,
    ):
        for chunk in response.iter_content(block_size):
            f.write(chunk)
            pbar.update(len(chunk))

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dataset_dir)

    os.remove(zip_path)


def load_fsdd_dataset(dataset_dir: str, max_files: Union[int, None] = None):
    recordings_dir = os.path.join(
        dataset_dir, "free-spoken-digit-dataset-master", "recordings"
    )
    audio_files, labels = [], []

    for filename in os.listdir(recordings_dir):
        if filename.endswith(".wav"):
            path = os.path.join(recordings_dir, filename)
            audio_files.append(path)
            labels.append(int(filename.split("_")[0]))
            if max_files and len(audio_files) >= max_files:
                break
    return audio_files, labels


def play_sample(audio_path: str):
    sr, data = wavfile.read(audio_path)
    data = data.astype(np.float32) / np.max(np.abs(data))
    sd.play(data, sr)
    sd.wait()


def main():
    dataset_dir = "fsdd"
    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
    recordings_dir = os.path.join(
        dataset_dir, "free-spoken-digit-dataset-master", "recordings"
    )

    if not os.path.exists(recordings_dir):
        download_and_extract_dataset(url, dataset_dir)

    audio_files, _ = load_fsdd_dataset(dataset_dir)
    print("Playing an audio sample")
    play_sample(audio_files[0])
    print("Playback completed..")


if __name__ == "__main__":
    main()
