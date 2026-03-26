"""Data loading utilities for lightning prediction tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def strikes_to_heatmap(strike_coords: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Convert strike coordinates to a per-pixel count map."""
    heatmap = np.zeros(shape, dtype=np.int32)
    if not strike_coords:
        return heatmap

    coords = np.array(strike_coords, dtype=np.float32)
    x_coords = np.clip(coords[:, 0].astype(np.int32), 0, shape[1] - 1)
    y_coords = np.clip(coords[:, 1].astype(np.int32), 0, shape[0] - 1)
    np.add.at(heatmap, (y_coords, x_coords), 1)
    return heatmap


def heatmap_to_coordinates(heatmap: np.ndarray) -> list[tuple[float, float]]:
    """Convert a count heatmap into repeated centroid coordinates."""
    coordinates: list[tuple[float, float]] = []
    y_indices, x_indices = np.nonzero(heatmap)
    counts = heatmap[y_indices, x_indices].astype(np.int32)

    for x_val, y_val, count in zip(x_indices, y_indices, counts):
        centroid = (float(x_val) + 0.5, float(y_val) + 0.5)
        coordinates.extend([centroid] * int(count))
    return coordinates


def frame_strike_coords(event: dict[str, np.ndarray], frame_number: int) -> list[tuple[float, float]]:
    """Get lightning coordinates for one 5-minute frame window."""
    if "lght" not in event:
        return []

    t = event["lght"][:, 0]
    window_center = frame_number * 5 * 60
    half_window = 2.5 * 60
    mask = (t >= window_center - half_window) & (t < window_center + half_window)
    return list(zip(event["lght"][mask, 3], event["lght"][mask, 4]))


def get_frame_data(event: dict[str, np.ndarray], frame_number: int) -> dict[str, Any]:
    """Retrieve channel slices and strike metadata for one frame."""
    output: dict[str, Any] = {}
    coords = frame_strike_coords(event, frame_number)
    output["lght_vil_coords"] = coords
    output["n_strikes"] = len(coords)

    for channel in ("vis", "ir069", "ir107", "vil"):
        output[channel] = event[channel][:, :, frame_number]
    return output


@dataclass(frozen=True)
class DataConfig:
    file_path: Path
    test_data: bool = False


class LightningFrameDataset(Dataset):
    """Frame-wise dataset with optional heatmap and count targets."""

    def __init__(
        self,
        event_ids: list[str],
        file_path: str | Path = "data/preprocessed_train.h5",
        test_data: bool = False,
    ) -> None:
        self.event_ids = event_ids
        self.frame_numbers = range(36)
        self.samples = [(event_id, frame) for event_id in event_ids for frame in self.frame_numbers]
        self.file_path = Path(file_path)
        self.test_data = test_data
        self._hdf5: h5py.File | None = None

    def _init_hdf5(self) -> None:
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.file_path, "r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._init_hdf5()
        assert self._hdf5 is not None

        event_id, frame_number = self.samples[idx]
        event_group = self._hdf5[event_id]

        event = {
            "vis": event_group["vis"][:],
            "ir069": event_group["ir069"][:],
            "ir107": event_group["ir107"][:],
            "vil": event_group["vil"][:],
        }
        if not self.test_data and "lght" in event_group:
            event["lght"] = event_group["lght"][:]

        frame_data = get_frame_data(event, frame_number)
        channels = [frame_data[name].astype(np.float32) for name in ("vis", "ir069", "ir107", "vil")]
        input_tensor = torch.tensor(np.stack(channels, axis=0), dtype=torch.float32)

        if self.test_data:
            return input_tensor

        height, width = frame_data["vis"].shape
        heatmap = strikes_to_heatmap(frame_data["lght_vil_coords"], shape=(height, width))
        binary_heatmap = torch.tensor((heatmap >= 1).astype(np.float32), dtype=torch.float32)
        count_target = torch.tensor([float(frame_data["n_strikes"])], dtype=torch.float32)
        targets = {"heatmap": binary_heatmap, "count": count_target}
        return input_tensor, targets


class LightningStrikeDataset(Dataset):
    """Event-wise sequence dataset for direct strike coordinate prediction."""

    def __init__(
        self,
        event_ids: list[str],
        file_path: str | Path = "data/preprocessed_train.h5",
        test_data: bool = False,
        max_strikes: int = 10000,
    ) -> None:
        self.event_ids = event_ids
        self.file_path = Path(file_path)
        self.test_data = test_data
        self.max_strikes = max_strikes if not test_data else 0
        self._hdf5: h5py.File | None = None

    def _init_hdf5(self) -> None:
        if self._hdf5 is None:
            self._hdf5 = h5py.File(self.file_path, "r")

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._init_hdf5()
        assert self._hdf5 is not None

        event_id = self.event_ids[idx]
        group = self._hdf5[event_id]

        ir069 = group["ir069"][:]
        vil = group["vil"][:]
        input_tensor = torch.tensor(np.stack([ir069, vil], axis=0), dtype=torch.float32)

        if self.test_data:
            return input_tensor

        if "lght" not in group:
            raise KeyError("Expected 'lght' dataset for non-test sequence training.")

        lght = group["lght"][:, [0, -2, -1]].astype(np.float32)
        lght_tensor = torch.tensor(lght, dtype=torch.float32)

        if lght_tensor.shape[0] < self.max_strikes:
            pad = torch.zeros((self.max_strikes - lght_tensor.shape[0], 3), dtype=torch.float32)
            lght_tensor = torch.cat([lght_tensor, pad], dim=0)
        elif lght_tensor.shape[0] > self.max_strikes:
            distances = torch.cdist(lght_tensor, lght_tensor, p=2)
            proximity = torch.sum(distances, dim=1)
            order = torch.argsort(proximity)
            lght_tensor = lght_tensor[order][: self.max_strikes]

        lght_tensor[:, 0] /= float(36 * 5 * 60)
        lght_tensor[:, 1] /= float(ir069.shape[1])
        lght_tensor[:, 2] /= float(ir069.shape[0])

        return input_tensor, lght_tensor


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Create a performant DataLoader with worker-safe defaults."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def list_event_ids(file_path: str | Path) -> list[str]:
    """Read event IDs from an HDF5 file."""
    with h5py.File(file_path, "r") as handle:
        return sorted(list(handle.keys()))
