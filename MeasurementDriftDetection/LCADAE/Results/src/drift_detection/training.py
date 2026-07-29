from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .io_utils import atomic_write_dataframe
from .proposed import lcad_loss, lcad_scores


def _loader(array: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    return DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_value: float,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_value": best_value,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train_reconstruction_model(
    model: nn.Module,
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    num_workers: int = 0,
    resume: bool = True,
) -> pd.DataFrame:
    checkpoint = Path(checkpoint_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    model.to(device)
    start_epoch = 1
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    if resume and checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_value = float(payload.get("best_value", best_value))
    train_loader = _loader(train_windows, batch_size, True, num_workers)
    validation_loader = _loader(validation_windows, batch_size, False, num_workers)
    stale_epochs = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for (batch,) in progress:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
            progress.set_postfix(loss=f"{np.mean(train_losses):.5f}")
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device, non_blocking=True)
                validation_losses.append(float(criterion(model(batch), batch).item()))
        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        atomic_write_dataframe(pd.DataFrame(history), history_path)
        if validation_loss < best_value:
            best_value = validation_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_checkpoint(checkpoint, model, optimizer, epoch, best_value)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state"])
    return pd.DataFrame(history)


def reconstruction_scores(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval().to(device)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in _loader(windows, batch_size, False):
            batch = batch.to(device, non_blocking=True)
            reconstruction = model(batch)
            batch_scores = ((batch - reconstruction) ** 2).mean(dim=(1, 2))
            scores.append(batch_scores.detach().cpu().numpy())
    return np.concatenate(scores)


def train_usad_model(
    model: nn.Module,
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    num_workers: int = 0,
) -> pd.DataFrame:
    model.to(device)
    optimizer1 = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer2 = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()
    train_loader = _loader(train_windows, batch_size, True, num_workers)
    validation_loader = _loader(validation_windows, batch_size, False, num_workers)
    history: list[dict[str, float]] = []
    best_value = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        losses1: list[float] = []
        losses2: list[float] = []
        coefficient = 1.0 / epoch
        progress = tqdm(train_loader, desc=f"USAD epoch {epoch}/{epochs}", leave=False)
        for (batch,) in progress:
            batch = batch.to(device, non_blocking=True)
            optimizer1.zero_grad(set_to_none=True)
            w1, _, w3 = model(batch)
            loss1 = coefficient * criterion(w1, batch) + (1.0 - coefficient) * criterion(w3, batch)
            loss1.backward()
            optimizer1.step()

            optimizer2.zero_grad(set_to_none=True)
            _, w2, w3 = model(batch)
            loss2 = coefficient * criterion(w2, batch) - (1.0 - coefficient) * criterion(w3, batch)
            loss2.backward()
            optimizer2.step()
            losses1.append(float(loss1.item()))
            losses2.append(float(loss2.item()))
            progress.set_postfix(loss1=f"{np.mean(losses1):.5f}", loss2=f"{np.mean(losses2):.5f}")
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device, non_blocking=True)
                w1, _, w3 = model(batch)
                validation_losses.append(float((0.5 * criterion(w1, batch) + 0.5 * criterion(w3, batch)).item()))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss_decoder1": float(np.mean(losses1)),
                "train_loss_decoder2": float(np.mean(losses2)),
                "validation_loss": validation_loss,
            }
        )
        atomic_write_dataframe(pd.DataFrame(history), history_path)
        if validation_loss < best_value:
            best_value = validation_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_checkpoint(Path(checkpoint_path), model, None, epoch, best_value)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def usad_scores(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
    alpha: float = 0.5,
) -> np.ndarray:
    model.eval().to(device)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in _loader(windows, batch_size, False):
            batch = batch.to(device, non_blocking=True)
            w1, _, w3 = model(batch)
            first = ((batch - w1) ** 2).mean(dim=(1, 2))
            third = ((batch - w3) ** 2).mean(dim=(1, 2))
            scores.append((alpha * first + (1.0 - alpha) * third).detach().cpu().numpy())
    return np.concatenate(scores)


def train_lcad_model(
    model: nn.Module,
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    loss_weights: dict[str, float],
    num_workers: int = 0,
) -> pd.DataFrame:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = _loader(train_windows, batch_size, True, num_workers)
    validation_loader = _loader(validation_windows, batch_size, False, num_workers)
    history: list[dict[str, float]] = []
    best_value = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        component_values = {"total": [], "reconstruction": [], "statistics": [], "correlation": []}
        progress = tqdm(train_loader, desc=f"LCAD-AE epoch {epoch}/{epochs}", leave=False)
        for (batch,) in progress:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, predicted_mean, predicted_log_std = model(batch)
            total, components = lcad_loss(
                batch,
                reconstruction,
                predicted_mean,
                predicted_log_std,
                reconstruction_weight=loss_weights["reconstruction"],
                statistics_weight=loss_weights["statistics"],
                correlation_weight=loss_weights["correlation"],
            )
            total.backward()
            optimizer.step()
            component_values["total"].append(float(total.item()))
            component_values["reconstruction"].append(float(components["reconstruction_loss"].item()))
            component_values["statistics"].append(float(components["statistics_loss"].item()))
            component_values["correlation"].append(float(components["correlation_loss"].item()))
            progress.set_postfix(loss=f"{np.mean(component_values['total']):.5f}")
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(device, non_blocking=True)
                outputs = model(batch)
                total, _ = lcad_loss(
                    batch,
                    *outputs,
                    reconstruction_weight=loss_weights["reconstruction"],
                    statistics_weight=loss_weights["statistics"],
                    correlation_weight=loss_weights["correlation"],
                )
                validation_losses.append(float(total.item()))
        validation_loss = float(np.mean(validation_losses))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(component_values["total"])),
            "train_reconstruction_loss": float(np.mean(component_values["reconstruction"])),
            "train_statistics_loss": float(np.mean(component_values["statistics"])),
            "train_correlation_loss": float(np.mean(component_values["correlation"])),
            "validation_loss": validation_loss,
        }
        history.append(row)
        atomic_write_dataframe(pd.DataFrame(history), history_path)
        if validation_loss < best_value:
            best_value = validation_loss
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            _save_checkpoint(Path(checkpoint_path), model, optimizer, epoch, best_value)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return pd.DataFrame(history)


def lcad_anomaly_scores(
    model: nn.Module,
    windows: np.ndarray,
    device: torch.device,
    score_weights: dict[str, float],
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    window_scores: list[np.ndarray] = []
    channel_scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in _loader(windows, batch_size, False):
            batch = batch.to(device, non_blocking=True)
            reconstruction, predicted_mean, predicted_log_std = model(batch)
            score, per_channel = lcad_scores(
                batch,
                reconstruction,
                predicted_mean,
                predicted_log_std,
                reconstruction_weight=score_weights["reconstruction"],
                statistics_weight=score_weights["statistics"],
                correlation_weight=score_weights["correlation"],
            )
            window_scores.append(score.detach().cpu().numpy())
            channel_scores.append(per_channel.detach().cpu().numpy())
    return np.concatenate(window_scores), np.concatenate(channel_scores)


def inference_latency_ms(
    scoring_function,
    windows: np.ndarray,
    repetitions: int = 3,
) -> float:
    sample = windows[: min(len(windows), 512)]
    if len(sample) == 0:
        return float("nan")
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        scoring_function(sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0 / len(sample))
    return float(np.median(timings))
