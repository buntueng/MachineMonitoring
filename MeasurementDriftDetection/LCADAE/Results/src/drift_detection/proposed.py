from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        groups = 4 if channels % 4 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return x + residual


class LCADAutoencoder(nn.Module):
    """Lightweight Context-Aware Dual-Scale Autoencoder.

    The local branch reconstructs each time step. The context head predicts the
    per-channel mean and log-standard-deviation of the full window.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 32,
        dilation_rates: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.input_projection = nn.Conv1d(n_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(
            *[
                DepthwiseSeparableTemporalBlock(hidden_dim, kernel_size, dilation, dropout)
                for dilation in dilation_rates
            ]
        )
        self.output_projection = nn.Conv1d(hidden_dim, n_features, kernel_size=1)
        self.statistics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        channels_first = x.transpose(1, 2)
        latent = self.blocks(self.input_projection(channels_first))
        reconstruction = self.output_projection(latent).transpose(1, 2)
        pooled = latent.mean(dim=2)
        statistics = self.statistics_head(pooled)
        predicted_mean, predicted_log_std = statistics.chunk(2, dim=1)
        return reconstruction, predicted_mean, predicted_log_std


def correlation_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = x - x.mean(dim=1, keepdim=True)
    scale = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    normalized = centered / scale
    return torch.matmul(normalized.transpose(1, 2), normalized) / max(1, x.shape[1])


def lcad_loss(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_log_std: torch.Tensor,
    reconstruction_weight: float = 1.0,
    statistics_weight: float = 0.25,
    correlation_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction_loss = F.mse_loss(reconstruction, x)
    true_mean = x.mean(dim=1)
    true_log_std = torch.log(x.std(dim=1, unbiased=False).clamp_min(1e-5))
    statistics_loss = F.mse_loss(predicted_mean, true_mean) + F.mse_loss(predicted_log_std, true_log_std)
    correlation_loss = F.l1_loss(correlation_matrix(reconstruction), correlation_matrix(x))
    total = (
        reconstruction_weight * reconstruction_loss
        + statistics_weight * statistics_loss
        + correlation_weight * correlation_loss
    )
    return total, {
        "reconstruction_loss": reconstruction_loss.detach(),
        "statistics_loss": statistics_loss.detach(),
        "correlation_loss": correlation_loss.detach(),
    }


def lcad_scores(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    predicted_mean: torch.Tensor,
    predicted_log_std: torch.Tensor,
    reconstruction_weight: float = 1.0,
    statistics_weight: float = 0.35,
    correlation_weight: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_channel_reconstruction = ((x - reconstruction) ** 2).mean(dim=1)
    true_mean = x.mean(dim=1)
    true_log_std = torch.log(x.std(dim=1, unbiased=False).clamp_min(1e-5))
    per_channel_statistics = (true_mean - predicted_mean) ** 2 + (true_log_std - predicted_log_std) ** 2
    correlation = torch.mean(torch.abs(correlation_matrix(x) - correlation_matrix(reconstruction)), dim=(1, 2))
    per_channel = reconstruction_weight * per_channel_reconstruction + statistics_weight * per_channel_statistics
    window_score = per_channel.mean(dim=1) + correlation_weight * correlation
    return window_score, per_channel
