from __future__ import annotations

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        input_dim = window_size * n_features
        middle = max(hidden_dim, latent_dim * 2)
        self.window_size = window_size
        self.n_features = n_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, middle),
            nn.ReLU(),
            nn.Linear(middle, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, middle),
            nn.ReLU(),
            nn.Linear(middle, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        reconstruction = self.decoder(self.encoder(flat))
        return reconstruction.reshape_as(x)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        repeated = self.from_latent(latent).unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, n_features, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        reconstruction = self.decoder(self.encoder(channels_first))
        return reconstruction.transpose(1, 2)


class USAD(nn.Module):
    """USAD-style shared encoder with two decoders."""

    def __init__(self, window_size: int, n_features: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        input_dim = window_size * n_features
        middle = max(hidden_dim, latent_dim * 2)
        self.window_size = window_size
        self.n_features = n_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, middle),
            nn.ReLU(),
            nn.Linear(middle, latent_dim),
            nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(nn.Linear(latent_dim, middle), nn.ReLU(), nn.Linear(middle, input_dim))
        self.decoder2 = nn.Sequential(nn.Linear(latent_dim, middle), nn.ReLU(), nn.Linear(middle, input_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(x.shape[0], -1)
        latent = self.encoder(flat)
        w1 = self.decoder1(latent)
        w2 = self.decoder2(latent)
        w3 = self.decoder2(self.encoder(w1))
        return w1.reshape_as(x), w2.reshape_as(x), w3.reshape_as(x)


def build_baseline_model(
    name: str,
    window_size: int,
    n_features: int,
    hidden_dim: int = 64,
    latent_dim: int = 16,
) -> nn.Module:
    normalized = name.lower()
    if normalized == "denseae":
        return DenseAutoencoder(window_size, n_features, hidden_dim, latent_dim)
    if normalized == "lstmae":
        return LSTMAutoencoder(n_features, hidden_dim, latent_dim)
    if normalized == "convae":
        return ConvolutionalAutoencoder(n_features, hidden_dim, latent_dim)
    if normalized == "usad":
        return USAD(window_size, n_features, hidden_dim, latent_dim)
    raise KeyError(f"Unsupported baseline model: {name}")
