import os
import torch
import numpy as np
from scipy.io import wavfile

from model.autoencoder import SimpleAutoencoder
from data.processor import AudioPreprocessor


def test_model(model_path, data_dir, output_dir="reconstructions"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = AudioPreprocessor()
    model = SimpleAutoencoder(input_size=preprocessor.target_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for file in os.listdir(data_dir):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(data_dir, file)
        audio = preprocessor.load_and_preprocess_audio(file_path)
        patches = preprocessor.extract_patches(audio)
        patches = preprocessor.apply_windowing(patches)
        patches = preprocessor.scaler.fit_transform(patches)  # normalize per file

        noisy = patches + np.random.normal(0, 0.05, patches.shape)
        noisy_tensor = torch.tensor(noisy, dtype=torch.float32).to(device)

        with torch.no_grad():
            recon = model(noisy_tensor).cpu().numpy()

        recon_audio = recon.flatten()
        recon_audio /= np.max(np.abs(recon_audio)) + 1e-8  # normalize for saving
        save_path = os.path.join(output_dir, f"reconstructed_{file}")
        wavfile.write(
            save_path, preprocessor.sample_rate, (recon_audio * 32767).astype(np.int16)
        )
        print(f"Reconstructed file saved: {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python test.py <model_path> <data_directory>")
        exit(1)
    model_path, data_dir = sys.argv[1], sys.argv[2]
    test_model(model_path, data_dir)
