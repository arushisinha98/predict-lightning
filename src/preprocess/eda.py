"""Run lean EDA for train.h5 and persist summary artifacts.

Usage:
    pixi run explore-train-h5
"""

import argparse
from pathlib import Path

from eda_utils import set_default_theme
from h5_eda_plots import (
    plot_channel_percentiles,
    plot_channel_presence,
    plot_event_type_distribution,
    plot_frame_intensity_trends,
    plot_lightning_hist,
    plot_lightning_frame_trend,
    plot_shape_consistency,
)
from h5_eda_utils import (
    build_schema_summary,
    collect_lightning_counts_per_event,
    compute_frame_intensity_trends,
    compute_channel_stats,
    compute_lightning_frame_trend,
    discover_event_ids,
    load_events_summary,
    resolve_events_csv_path,
    resolve_train_h5_path,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures" / "h5_eda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HDF5 EDA on train.h5 and save summary tables and figures."
    )
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=REPO_ROOT / "data" / "train.h5",
        help="Path to input HDF5 file (default: %(default)s).",
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=REPO_ROOT / "data" / "events.csv",
        help="Path to events CSV for metadata joins (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where EDA artifacts are written (default: %(default)s).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=200,
        help="Maximum number of events to scan/sample (default: %(default)s).",
    )
    parser.add_argument(
        "--max-points-per-event",
        type=int,
        default=15000,
        help="Per-event sample cap for numeric stats (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_default_theme()

    input_h5 = resolve_train_h5_path(REPO_ROOT, args.input_h5)
    events_csv = resolve_events_csv_path(REPO_ROOT, args.events_csv)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_h5.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_h5}")
    if args.max_events <= 0:
        raise ValueError("--max-events must be > 0")
    if args.max_points_per_event <= 0:
        raise ValueError("--max-points-per-event must be > 0")

    all_event_ids = discover_event_ids(input_h5)
    if not all_event_ids:
        raise RuntimeError(f"No event groups found in {input_h5}")

    print(f"Input HDF5: {input_h5}")
    print(f"Total events in file: {len(all_event_ids)}")
    print(f"Events scanned: {min(args.max_events, len(all_event_ids))}")

    schema_df = build_schema_summary(input_h5, max_events=args.max_events)
    stats_df = compute_channel_stats(
        input_h5,
        max_events=args.max_events,
        max_points_per_event=args.max_points_per_event,
        seed=args.seed,
    )
    lightning_df = collect_lightning_counts_per_event(input_h5, max_events=args.max_events)
    frame_intensity_df = compute_frame_intensity_trends(
        input_h5,
        max_events=args.max_events,
    )
    lightning_frame_df = compute_lightning_frame_trend(
        input_h5,
        max_events=args.max_events,
    )

    schema_csv = output_dir / "h5_schema_summary.csv"
    stats_csv = output_dir / "h5_channel_stats.csv"
    lightning_csv = output_dir / "h5_lightning_summary.csv"
    frame_intensity_csv = output_dir / "h5_frame_intensity_trends.csv"
    lightning_frame_csv = output_dir / "h5_lightning_frame_trend.csv"

    schema_df.to_csv(schema_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    lightning_df.to_csv(lightning_csv, index=False)
    frame_intensity_df.to_csv(frame_intensity_csv, index=False)
    lightning_frame_df.to_csv(lightning_frame_csv, index=False)

    saved_plots = [
        plot_channel_presence(schema_df, output_dir),
        plot_shape_consistency(schema_df, output_dir),
        plot_channel_percentiles(stats_df, output_dir),
    ]

    lght_plot = plot_lightning_hist(lightning_df, output_dir)
    if lght_plot is not None:
        saved_plots.append(lght_plot)

    frame_plot = plot_frame_intensity_trends(frame_intensity_df, output_dir)
    if frame_plot is not None:
        saved_plots.append(frame_plot)

    frame_lght_plot = plot_lightning_frame_trend(lightning_frame_df, output_dir)
    if frame_lght_plot is not None:
        saved_plots.append(frame_lght_plot)

    event_summary_csv = None
    if events_csv.exists():
        candidate_ids = set(all_event_ids[: min(args.max_events, len(all_event_ids))])
        event_type_df = load_events_summary(events_csv, candidate_ids=candidate_ids)
        event_summary_csv = output_dir / "h5_event_type_summary.csv"
        event_type_df.to_csv(event_summary_csv, index=False)

        event_plot = plot_event_type_distribution(event_type_df, output_dir)
        if event_plot is not None:
            saved_plots.append(event_plot)
    else:
        print(f"Warning: events CSV not found, skipping metadata join: {events_csv}")

    print("Saved tables:")
    print(f"- {schema_csv}")
    print(f"- {stats_csv}")
    print(f"- {lightning_csv}")
    print(f"- {frame_intensity_csv}")
    print(f"- {lightning_frame_csv}")
    if event_summary_csv is not None:
        print(f"- {event_summary_csv}")

    print("Saved figures:")
    for path in saved_plots:
        print(f"- {path}")


if __name__ == "__main__":
    main()
