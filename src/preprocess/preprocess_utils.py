"""Utilities for HDF5 event loading, resizing, normalization, and stats."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import zoom


IMAGE_CHANNELS = ("vis", "ir069", "ir107", "vil")


@dataclass(frozen=True)
class ResizeConfig:
    direction: str = "down"
    up_size: int = 384
    down_size: int = 192

    @property
    def target_size(self) -> int:
        if self.direction == "up":
            return self.up_size
        if self.direction == "down":
            return self.down_size
        raise ValueError("direction must be one of {'up', 'down'}")

    @property
    def interpolation_order(self) -> int:
        # Bicubic-like interpolation for upscaling, linear for downscaling.
        return 3 if self.direction == "up" else 1


def load_event_from_group(event_group: h5py.Group, include_lightning: bool = True) -> dict[str, np.ndarray]:
    """Load one event group into memory."""
    channels = list(IMAGE_CHANNELS)
    if include_lightning and "lght" in event_group:
        channels.append("lght")
    return {channel: event_group[channel][:] for channel in channels if channel in event_group}


def resize_channel_stack(channel_stack: np.ndarray, target_size: int, order: int) -> np.ndarray:
    """Resize an H x W x T channel stack to target_size x target_size x T."""
    if channel_stack.ndim != 3:
        raise ValueError("Expected channel stack with shape (H, W, T)")

    height, width, _ = channel_stack.shape
    if height == target_size and width == target_size:
        return channel_stack.astype(np.float32)

    scale_h = target_size / float(height)
    scale_w = target_size / float(width)
    resized = zoom(channel_stack, (scale_h, scale_w, 1.0), order=order)
    return resized.astype(np.float32)


def min_max_scale(channel_stack: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a stack to [0, 1], handling flat arrays safely."""
    min_val = float(np.min(channel_stack))
    max_val = float(np.max(channel_stack))
    scale = max_val - min_val
    if scale < eps:
        return np.zeros_like(channel_stack, dtype=np.float32)
    return ((channel_stack - min_val) / scale).astype(np.float32)


def preprocess_event(
    event: dict[str, np.ndarray],
    resize_cfg: ResizeConfig,
    normalize_image_channels: bool = True,
) -> dict[str, np.ndarray]:
    """Resize and normalize a single event dictionary."""
    processed: dict[str, np.ndarray] = {}

    for channel, array in event.items():
        if channel in IMAGE_CHANNELS:
            resized = resize_channel_stack(
                array,
                target_size=resize_cfg.target_size,
                order=resize_cfg.interpolation_order,
            )
            processed[channel] = min_max_scale(resized) if normalize_image_channels else resized
        else:
            processed[channel] = array

    return processed


def compute_channel_stats(channel_stack: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for one channel stack."""
    return {
        "min": float(np.min(channel_stack)),
        "max": float(np.max(channel_stack)),
        "mean": float(np.mean(channel_stack)),
        "std": float(np.std(channel_stack)),
        "p01": float(np.percentile(channel_stack, 1)),
        "p99": float(np.percentile(channel_stack, 99)),
        "dynamic_range": float(np.max(channel_stack) - np.min(channel_stack)),
    }


def resolve_h5_path(repo_root: Path, user_path: Path | None, default_name: str) -> Path:
    """Resolve optional user path against the repository root."""
    if user_path is None:
        return repo_root / "data" / default_name
    return user_path if user_path.is_absolute() else (repo_root / user_path)
