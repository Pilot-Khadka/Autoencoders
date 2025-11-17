import torch
import torch.nn as nn
from typing import Union
import torch.nn.functional as F


class ConvAutoencoder(nn.Module):
    def __init__(self, n_mels: int = 128, latent_channels: int = 16):
        super(ConvAutoencoder, self).__init__()
        self.n_mels = n_mels

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        # convtranspose 2d also called  deconvolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_mels, time_frames = x.shape
        x = x.unsqueeze(1)

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded = decoded.squeeze(1)

        if decoded.shape[-2:] != (n_mels, time_frames):
            decoded = F.interpolate(
                decoded.unsqueeze(1),
                size=(n_mels, time_frames),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return decoded

    def training_step(
        self, x_clean: torch.Tensor, x_noisy: torch.Tensor, optimizer
    ) -> Union[float, int]:
        optimizer.zero_grad()
        output = self.forward(x_noisy)
        loss = F.mse_loss(output, x_clean)
        loss.backward()
        optimizer.step()
        return loss.item()


class SkipConvAutoencoder(nn.Module):
    """Unet like architecture"""

    def __init__(self, n_mels: int = 128, latent_channels: int = 16):
        super(SkipConvAutoencoder, self).__init__()
        self.n_mels = n_mels

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.dec4 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_mels, time_frames = x.shape
        x = x.unsqueeze(1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.dec1(e4)
        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)

        decoded = self.dec4(d3)
        decoded = decoded.squeeze(1)

        if decoded.shape[-2:] != (n_mels, time_frames):
            decoded = F.interpolate(
                decoded.unsqueeze(1),
                size=(n_mels, time_frames),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return decoded

    def training_step(
        self, x_clean: torch.Tensor, x_noisy: torch.Tensor, optimizer
    ) -> Union[float, int]:
        optimizer.zero_grad()
        output = self.forward(x_noisy)
        loss = F.mse_loss(output, x_clean)
        loss.backward()
        optimizer.step()
        return loss.item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvAutoencoder(n_mels=128, latent_channels=16).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    clean_mel = torch.randn(8, 128, 100).to(device)
    noisy_mel = clean_mel + torch.randn_like(clean_mel) * 0.1

    model.train()
    loss = model.training_step(clean_mel, noisy_mel, optimizer)
    print(f"Training loss: {loss:.4f}")

    model.eval()
    with torch.no_grad():
        denoised = model(noisy_mel)
    print(f"Output shape: {denoised.shape}")


if __name__ == "__main__":
    main()
