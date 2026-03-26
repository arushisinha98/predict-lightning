"""Loss registry for configurable training objectives."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_outputs(outputs: object) -> dict[str, torch.Tensor]:
    """Normalize output structure to a named dictionary."""
    if isinstance(outputs, dict):
        return outputs
    if isinstance(outputs, tuple):
        if len(outputs) == 2:
            return {"heatmap": outputs[0], "count": outputs[1]}
        if len(outputs) == 1:
            return {"heatmap": outputs[0]}
    if torch.is_tensor(outputs):
        return {"heatmap": outputs}
    raise TypeError("Unsupported model output type for loss computation.")


def unpack_targets(targets: object) -> dict[str, torch.Tensor]:
    """Normalize target structure to a named dictionary."""
    if isinstance(targets, dict):
        return targets
    if torch.is_tensor(targets):
        return {"heatmap": targets}
    raise TypeError("Unsupported target type for loss computation.")


class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation masks."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        intersection = torch.sum(pred_flat * target_flat, dim=1)
        denom = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEHeatmapLoss(nn.Module):
    """Binary cross-entropy on heatmap output."""

    def forward(self, outputs: object, targets: object) -> torch.Tensor:
        out = unpack_outputs(outputs)
        tgt = unpack_targets(targets)
        return F.binary_cross_entropy(out["heatmap"], tgt["heatmap"])


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice objective for heatmap segmentation."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, outputs: object, targets: object) -> torch.Tensor:
        out = unpack_outputs(outputs)
        tgt = unpack_targets(targets)
        bce = F.binary_cross_entropy(out["heatmap"], tgt["heatmap"])
        dice = self.dice(out["heatmap"], tgt["heatmap"])
        return self.bce_weight * bce + self.dice_weight * dice


class MultiTaskHeatmapCountLoss(nn.Module):
    """Joint heatmap BCE + count MSE for dual-head models."""

    def __init__(self, heatmap_weight: float = 1.0, count_weight: float = 0.1) -> None:
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.count_weight = count_weight

    def forward(self, outputs: object, targets: object) -> torch.Tensor:
        out = unpack_outputs(outputs)
        tgt = unpack_targets(targets)

        heatmap_loss = F.binary_cross_entropy(out["heatmap"], tgt["heatmap"])
        if "count" not in out or "count" not in tgt:
            return self.heatmap_weight * heatmap_loss

        pred_count = out["count"].reshape(-1)
        target_count = tgt["count"].reshape(-1)
        count_loss = F.mse_loss(pred_count, target_count)
        return self.heatmap_weight * heatmap_loss + self.count_weight * count_loss


LossFactory = Callable[..., nn.Module]


LOSS_REGISTRY: dict[str, LossFactory] = {
    "bce": lambda **_: BCEHeatmapLoss(),
    "dice": lambda **_: DiceLossWrapper(),
    "bce-dice": lambda **_: BCEDiceLoss(),
    "multitask": lambda **_: MultiTaskHeatmapCountLoss(),
}


class DiceLossWrapper(nn.Module):
    """Apply Dice loss over unpacked heatmap outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.dice = DiceLoss()

    def forward(self, outputs: object, targets: object) -> torch.Tensor:
        out = unpack_outputs(outputs)
        tgt = unpack_targets(targets)
        return self.dice(out["heatmap"], tgt["heatmap"])


def list_losses() -> list[str]:
    return sorted(LOSS_REGISTRY.keys())


def build_loss(name: str, **kwargs: object) -> nn.Module:
    """Instantiate a loss by registry name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list_losses()}")
    return LOSS_REGISTRY[name](**kwargs)
