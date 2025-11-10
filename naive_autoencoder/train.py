import torch
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from model.autoencoder import SimpleAutoencoder
from data.processor import AudioPreprocessor, AudioDataset


def train_model(
    data_dir, epochs=80, batch_size=32, learning_rate=0.001, save_path="autoencoder.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessor = AudioPreprocessor()
    dataset = AudioDataset(data_dir, preprocessor)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = SimpleAutoencoder(input_size=preprocessor.target_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=80)
        for noisy, clean in progress_bar:
            noisy, clean = noisy.to(device), clean.to(device)
            loss = model.training_step(clean, noisy, optimizer)
            total_loss += loss * noisy.size(0)
            progress_bar.set_postfix({"batch_loss": f"{loss:.6f}"})

        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train.py <data_directory>")
        exit(1)
    data_dir = sys.argv[1]
    train_model(data_dir)
