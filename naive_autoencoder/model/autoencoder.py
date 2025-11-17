import torch
import torch.nn as nn
from typing import Union, List
import torch.nn.functional as F


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [512, 256, 128, 64]):
        super(SimpleAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # encoder
        encoder_layers = []
        prev_size = input_size
        for h in hidden_sizes:
            encoder_layers.append(nn.Linear(prev_size, h))
            prev_size = h
        self.encoder = nn.ModuleList(encoder_layers)

        # decoder
        decoder_layers = []
        decoder_sizes = hidden_sizes[::-1] + [input_size]
        prev_size = decoder_sizes[0]
        for h in decoder_sizes[1:]:
            decoder_layers.append(nn.Linear(prev_size, h))
            prev_size = h
        self.decoder = nn.ModuleList(decoder_layers)

    def forward(self, x: torch.Tensor):
        # encoder
        for layer in self.encoder:
            x = torch.tanh(layer(x))

        # decoder
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:
                x = torch.tanh(x)
        return x

    def training_step(
        self, x_clean: torch.Tensor, x_noisy: torch.Tensor, optimizer
    ) -> Union[float, int]:
        optimizer.zero_grad()
        output = self.forward(x_noisy)
        loss = F.mse_loss(output, x_clean)
        loss.backward()
        optimizer.step()
        return loss.item()
