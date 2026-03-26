"""Model registry for plug-and-play architecture selection."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from unet import UNET


ModelFactory = Callable[..., nn.Module]


class HeatmapOnlyAdapter(nn.Module):
    """Wrap a model and return only heatmap output when model returns tuples."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, tuple):
            return output[0]
        if isinstance(output, dict) and "heatmap" in output:
            return output["heatmap"]
        return output


class TinyConvNet(nn.Module):
    """Minimal heatmap segmentation baseline."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _build_unet(device: str, **_: object) -> nn.Module:
    return UNET(device=device)


def _build_unet_heatmap(device: str, **_: object) -> nn.Module:
    return HeatmapOnlyAdapter(UNET(device=device))


def _build_tiny(device: str, **_: object) -> nn.Module:
    del device
    return TinyConvNet(in_channels=4)


MODEL_REGISTRY: dict[str, ModelFactory] = {
    "unet": _build_unet,
    "unet-heatmap": _build_unet_heatmap,
    "tiny-conv": _build_tiny,
}


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())


def build_model(name: str, device: str, **kwargs: object) -> nn.Module:
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list_models()}")
    return MODEL_REGISTRY[name](device=device, **kwargs)
