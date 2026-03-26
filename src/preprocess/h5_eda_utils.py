"""Utilities for lean EDA over HDF5 storm-event datasets."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


DEFAULT_CHANNELS = ("vis", "ir069", "ir107", "vil", "lght")


def _resolve_path(repo_root: Path, user_path: Path | None, fallback: str) -> Path:
    if user_path is None:
        return repo_root / "data" / fallback
    return user_path if user_path.is_absolute() else (repo_root / user_path)


def resolve_train_h5_path(repo_root: Path, user_path: Path | None = None) -> Path:
    return _resolve_path(repo_root, user_path, "train.h5")


def resolve_events_csv_path(repo_root: Path, user_path: Path | None = None) -> Path:
    return _resolve_path(repo_root, user_path, "events.csv")


def discover_event_ids(h5_path: Path) -> list[str]:
    with h5py.File(h5_path, "r") as h5f:
        return list(h5f.keys())


def _dataset_slice_for_stats(dataset: h5py.Dataset) -> np.ndarray:
    if dataset.ndim >= 3:
        return dataset[:, :, 0]
    if dataset.ndim == 2:
        return dataset[: min(2000, dataset.shape[0]), :]
    if dataset.ndim == 1:
        return dataset[: min(2000, dataset.shape[0])]
    return dataset[()]


def build_schema_summary(
    h5_path: Path,
    max_events: int,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5f:
        event_ids = list(h5f.keys())
        total_events = len(event_ids)
        scan_ids = event_ids[: min(max_events, total_events)]

        for channel in channels:
            present = 0
            shapes: set[tuple[int, ...]] = set()
            dtypes: set[str] = set()
            first_min: float | None = None
            first_max: float | None = None
            finite_ok = True
            non_finite_values = 0

            for event_id in scan_ids:
                group = h5f[event_id]
                if channel not in group:
                    continue

                present += 1
                dataset = group[channel]
                shapes.add(tuple(dataset.shape))
                dtypes.add(str(dataset.dtype))

                sample = _dataset_slice_for_stats(dataset)
                arr = np.asarray(sample)

                if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                    if first_min is None:
                        first_min = float(np.min(arr))
                        first_max = float(np.max(arr))
                    is_finite = np.isfinite(arr)
                    if not np.all(is_finite):
                        finite_ok = False
                        non_finite_values += int(arr.size - np.count_nonzero(is_finite))

            rows.append(
                {
                    "channel": channel,
                    "events_scanned": len(scan_ids),
                    "events_present": present,
                    "presence_ratio": (present / len(scan_ids)) if scan_ids else 0.0,
                    "shape_variants": len(shapes),
                    "dtype_variants": len(dtypes),
                    "shapes": "; ".join(str(s) for s in sorted(shapes)),
                    "dtypes": "; ".join(sorted(dtypes)),
                    "first_sample_min": first_min,
                    "first_sample_max": first_max,
                    "all_finite": finite_ok,
                    "non_finite_values": non_finite_values,
                    "shape_consistent": len(shapes) <= 1,
                    "dtype_consistent": len(dtypes) <= 1,
                    "total_events": total_events,
                }
            )

    return pd.DataFrame(rows)


def compute_channel_stats(
    h5_path: Path,
    max_events: int,
    max_points_per_event: int,
    seed: int,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5f:
        event_ids = list(h5f.keys())
        scan_ids = event_ids[: min(max_events, len(event_ids))]

        for channel in channels:
            samples: list[np.ndarray] = []
            scanned = 0
            for event_id in scan_ids:
                group = h5f[event_id]
                if channel not in group:
                    continue

                arr = np.asarray(group[channel][:]).reshape(-1)
                if arr.size == 0:
                    continue
                scanned += 1

                take = min(max_points_per_event, arr.size)
                if take < arr.size:
                    idx = rng.choice(arr.size, size=take, replace=False)
                    sample = arr[idx]
                else:
                    sample = arr
                samples.append(sample.astype(np.float64, copy=False))

            if not samples:
                rows.append(
                    {
                        "channel": channel,
                        "events_used": scanned,
                        "sample_points": 0,
                        "min": np.nan,
                        "max": np.nan,
                        "mean": np.nan,
                        "std": np.nan,
                        "p01": np.nan,
                        "p50": np.nan,
                        "p99": np.nan,
                    }
                )
                continue

            merged = np.concatenate(samples)
            rows.append(
                {
                    "channel": channel,
                    "events_used": scanned,
                    "sample_points": int(merged.size),
                    "min": float(np.min(merged)),
                    "max": float(np.max(merged)),
                    "mean": float(np.mean(merged)),
                    "std": float(np.std(merged)),
                    "p01": float(np.percentile(merged, 1)),
                    "p50": float(np.percentile(merged, 50)),
                    "p99": float(np.percentile(merged, 99)),
                }
            )

    return pd.DataFrame(rows)


def collect_lightning_counts_per_event(h5_path: Path, max_events: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5f:
        event_ids = list(h5f.keys())
        for event_id in event_ids[: min(max_events, len(event_ids))]:
            group = h5f[event_id]
            if "lght" not in group:
                rows.append({"id": event_id, "lightning_points": 0})
                continue
            rows.append({"id": event_id, "lightning_points": int(group["lght"].shape[0])})
    return pd.DataFrame(rows)


def compute_frame_intensity_trends(
    h5_path: Path,
    max_events: int,
    image_channels: tuple[str, ...] = ("vis", "ir069", "ir107", "vil"),
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5f:
        event_ids = list(h5f.keys())[:max_events]

        for channel in image_channels:
            per_event_means: list[np.ndarray] = []
            for event_id in event_ids:
                group = h5f[event_id]
                if channel not in group:
                    continue

                arr = np.asarray(group[channel][:])
                if arr.ndim != 3 or arr.size == 0:
                    continue

                frame_means = arr.reshape(-1, arr.shape[2]).mean(axis=0)
                per_event_means.append(frame_means.astype(np.float64, copy=False))

            if not per_event_means:
                continue

            stacked = np.vstack(per_event_means)
            for frame_idx in range(stacked.shape[1]):
                frame_values = stacked[:, frame_idx]
                rows.append(
                    {
                        "channel": channel,
                        "frame": int(frame_idx),
                        "event_count": int(stacked.shape[0]),
                        "mean_intensity": float(np.mean(frame_values)),
                        "std_intensity": float(np.std(frame_values)),
                        "p10_intensity": float(np.percentile(frame_values, 10)),
                        "p50_intensity": float(np.percentile(frame_values, 50)),
                        "p90_intensity": float(np.percentile(frame_values, 90)),
                    }
                )

    return pd.DataFrame(rows)


def compute_lightning_frame_trend(
    h5_path: Path,
    max_events: int,
    n_frames: int = 36,
    frame_interval_seconds: float = 300.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as h5f:
        event_ids = list(h5f.keys())[:max_events]
        per_event_counts: list[np.ndarray] = []

        frame_centers = np.arange(n_frames, dtype=np.float64) * frame_interval_seconds
        bin_edges = np.concatenate(([frame_centers[0] - frame_interval_seconds / 2.0], frame_centers + frame_interval_seconds / 2.0))

        for event_id in event_ids:
            group = h5f[event_id]
            if "lght" not in group:
                continue

            lght = np.asarray(group["lght"][:])
            if lght.ndim != 2 or lght.shape[0] == 0:
                per_event_counts.append(np.zeros(n_frames, dtype=np.int32))
                continue

            times = lght[:, 0]
            counts, _ = np.histogram(times, bins=bin_edges)
            per_event_counts.append(counts.astype(np.int32, copy=False))

        if not per_event_counts:
            return pd.DataFrame(rows)

        stacked = np.vstack(per_event_counts)
        for frame_idx in range(n_frames):
            frame_counts = stacked[:, frame_idx]
            rows.append(
                {
                    "frame": int(frame_idx),
                    "event_count": int(stacked.shape[0]),
                    "mean_lightning_count": float(np.mean(frame_counts)),
                    "p50_lightning_count": float(np.percentile(frame_counts, 50)),
                    "p90_lightning_count": float(np.percentile(frame_counts, 90)),
                    "max_lightning_count": int(np.max(frame_counts)),
                }
            )

    return pd.DataFrame(rows)


def load_events_summary(events_csv_path: Path, candidate_ids: set[str] | None = None) -> pd.DataFrame:
    events_df = pd.read_csv(events_csv_path)
    if "id" not in events_df.columns:
        raise ValueError("events.csv must include an 'id' column")

    dedup = events_df.drop_duplicates(subset=["id"])
    if candidate_ids is not None:
        dedup = dedup[dedup["id"].isin(candidate_ids)]

    summary = (
        dedup["event_type"]
        .value_counts(dropna=False)
        .rename_axis("event_type")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["share"] = summary["count"] / summary["count"].sum()
    return summary
