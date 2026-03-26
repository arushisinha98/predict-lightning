"""Reusable training and evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from losses import unpack_outputs, unpack_targets


@dataclass
class EpochMetrics:
    loss: float
    iou: float


def _move_targets_to_device(targets: object, device: str) -> object:
    if isinstance(targets, dict):
        return {k: v.to(device, non_blocking=True) for k, v in targets.items()}
    if torch.is_tensor(targets):
        return targets.to(device, non_blocking=True)
    return targets


def _compute_iou(outputs: object, targets: object, threshold: float = 0.5) -> float:
    out = unpack_outputs(outputs)
    tgt = unpack_targets(targets)
    if "heatmap" not in out or "heatmap" not in tgt:
        return 0.0

    pred = (out["heatmap"] >= threshold).to(torch.float32)
    truth = (tgt["heatmap"] >= threshold).to(torch.float32)

    pred = pred.reshape(pred.shape[0], -1)
    truth = truth.reshape(truth.shape[0], -1)
    intersection = torch.sum(pred * truth, dim=1)
    union = torch.sum(pred, dim=1) + torch.sum(truth, dim=1) - intersection
    iou = torch.where(union > 0, intersection / union, torch.ones_like(union))
    return float(iou.mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> EpochMetrics:
    model.train()
    loss_sum = 0.0
    iou_sum = 0.0
    num_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = _move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item())
        iou_sum += _compute_iou(outputs, targets)
        num_batches += 1

    denom = max(1, num_batches)
    return EpochMetrics(loss=loss_sum / denom, iou=iou_sum / denom)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> EpochMetrics:
    model.eval()
    loss_sum = 0.0
    iou_sum = 0.0
    num_batches = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = _move_targets_to_device(targets, device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        loss_sum += float(loss.item())
        iou_sum += _compute_iou(outputs, targets)
        num_batches += 1

    denom = max(1, num_batches)
    return EpochMetrics(loss=loss_sum / denom, iou=iou_sum / denom)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    checkpoint_dir: Path,
) -> None:
    """Run end-to-end training with best-checkpoint persistence."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} train_iou={train_metrics.iou:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_iou={val_metrics.iou:.4f}"
        )

        latest_path = checkpoint_dir / "last.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, latest_path)

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            best_path = checkpoint_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_metrics.loss,
                    "val_iou": val_metrics.iou,
                },
                best_path,
            )
