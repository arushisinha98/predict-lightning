"""Preprocess HDF5 storm events and visualize transformation diagnostics.

Usage:
    pixi run preprocess-hdf5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from eda_utils import save_figure, set_default_theme
from preprocess_utils import (
    IMAGE_CHANNELS,
    ResizeConfig,
    compute_channel_stats,
    load_event_from_group,
    preprocess_event,
    resolve_h5_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess HDF5 event data and generate visual diagnostics."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=REPO_ROOT / "data" / "train.h5",
        help="Input HDF5 file path.",
    )
    parser.add_argument(
        "--output-h5",
        type=Path,
        default=REPO_ROOT / "data" / "preprocessed_train.h5",
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "data" / "preprocess_summary.csv",
        help="CSV path for per-channel preprocessing statistics.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "preprocess",
        help="Directory for preprocessing diagnostic plots.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=100,
        help="Maximum number of events to preprocess; use -1 to process all events.",
    )
    parser.add_argument(
        "--direction",
        choices=["up", "down"],
        default="down",
        help="Resize direction for image channels.",
    )
    return parser.parse_args()


def plot_sample_before_after(
    raw_event: dict[str, np.ndarray],
    processed_event: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Show midpoint frame before and after preprocessing per channel."""
    fig, axes = plt.subplots(len(IMAGE_CHANNELS), 2, figsize=(10, 3 * len(IMAGE_CHANNELS)))
    fig.suptitle("Sample Event: Before vs After Preprocessing", y=1.01)

    for row, channel in enumerate(IMAGE_CHANNELS):
        raw_stack = raw_event[channel]
        proc_stack = processed_event[channel]

        raw_mid = raw_stack[:, :, raw_stack.shape[2] // 2]
        proc_mid = proc_stack[:, :, proc_stack.shape[2] // 2]

        axes[row, 0].imshow(raw_mid, cmap="gray")
        axes[row, 0].set_title(f"{channel} before")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(proc_mid, cmap="gray")
        axes[row, 1].set_title(f"{channel} after")
        axes[row, 1].axis("off")

    return save_figure(fig, output_dir, "preprocess_before_after_sample.png")


def plot_strikes_per_frame(strike_counts: list[int], output_dir: Path) -> Path | None:
    """Plot frame-level lightning strike count distribution if available."""
    if not strike_counts:
        return None

    strike_arr = np.array(strike_counts, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(strike_arr, bins=50, edgecolor="black", color=sns.color_palette("colorblind", 8)[0])
    ax.set_title("Distribution of Lightning Strikes per Frame")
    ax.set_xlabel("Strikes in Frame")
    ax.set_ylabel("Frequency")

    mean_val = float(np.mean(strike_arr))
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1)
    ax.text(
        mean_val * 1.03,
        ax.get_ylim()[1] * 0.9,
        f"mean={mean_val:.1f}\nmax={int(np.max(strike_arr))}",
    )

    return save_figure(fig, output_dir, "preprocess_lightning_strikes_per_frame.png")


def preprocess_events(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    """Preprocess events into a new HDF5 and collect diagnostics."""
    input_h5 = resolve_h5_path(REPO_ROOT, args.input_h5, "train.h5")
    output_h5 = resolve_h5_path(REPO_ROOT, args.output_h5, "preprocessed_train.h5")
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    resize_cfg = ResizeConfig(direction=args.direction)
    stats_rows: list[dict[str, object]] = []
    strike_counts_per_frame: list[int] = []

    sample_raw: dict[str, np.ndarray] | None = None
    sample_processed: dict[str, np.ndarray] | None = None

    with h5py.File(input_h5, "r") as f_in, h5py.File(output_h5, "w") as f_out:
        all_event_ids = list(f_in.keys())
        if args.max_events < 0:
            event_ids = all_event_ids
        else:
            event_ids = all_event_ids[: args.max_events]

        for event_id in tqdm(event_ids, desc="Preprocessing events"):
            raw_event = load_event_from_group(f_in[event_id])
            processed_event = preprocess_event(raw_event, resize_cfg=resize_cfg)

            if sample_raw is None:
                sample_raw = {k: v.copy() for k, v in raw_event.items() if k in IMAGE_CHANNELS}
                sample_processed = {k: v.copy() for k, v in processed_event.items() if k in IMAGE_CHANNELS}

            grp = f_out.create_group(event_id)
            for channel, data in processed_event.items():
                grp.create_dataset(channel, data=data, compression="lzf")

            for channel in IMAGE_CHANNELS:
                before_stats = compute_channel_stats(raw_event[channel])
                after_stats = compute_channel_stats(processed_event[channel])

                stats_rows.append({"event_id": event_id, "channel": channel, "stage": "before", **before_stats})
                stats_rows.append({"event_id": event_id, "channel": channel, "stage": "after", **after_stats})

            if "lght" in raw_event and raw_event["lght"].size > 0:
                strike_times = raw_event["lght"][:, 0]
                frame_centers = np.arange(36) * 300.0
                bin_edges = np.concatenate(([frame_centers[0] - 150.0], frame_centers + 150.0))
                counts, _ = np.histogram(strike_times, bins=bin_edges)
                strike_counts_per_frame.extend(counts.tolist())

    if sample_raw is None or sample_processed is None:
        raise RuntimeError("No events were processed. Check input file and --max-events.")

    return pd.DataFrame(stats_rows), sample_raw, sample_processed, strike_counts_per_frame


def main() -> None:
    args = parse_args()
    set_default_theme()

    print(f"Input H5: {args.input_h5}")
    print(f"Output H5: {args.output_h5}")
    print(f"Direction: {args.direction}")
    print(f"Max events: {args.max_events}")

    stats_df, sample_raw, sample_processed, strike_counts = preprocess_events(args)

    summary_csv = resolve_h5_path(REPO_ROOT, args.summary_csv, "preprocess_summary.csv")
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(summary_csv, index=False)

    figures_dir = args.figures_dir if args.figures_dir.is_absolute() else (REPO_ROOT / args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved = [
        plot_sample_before_after(sample_raw, sample_processed, figures_dir),
    ]
    strike_plot = plot_strikes_per_frame(strike_counts, figures_dir)
    if strike_plot is not None:
        saved.append(strike_plot)

    print(f"Saved preprocessing summary CSV: {summary_csv}")
    print("Saved preprocessing figures:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
