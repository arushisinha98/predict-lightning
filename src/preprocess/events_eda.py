"""Run exploratory data analysis for the events CSV and save figures.

Usage:
	pixi run explore-data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from eda_utils import (
	build_event_markers,
	build_event_palette,
	save_figure,
	set_default_theme,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS_CSV = REPO_ROOT / "data" / "events.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA on the events CSV and save summary charts to a folder."
    )
    parser.add_argument(
        "--events-csv",
        type=Path,
        default=DEFAULT_EVENTS_CSV,
        help="Path to events CSV file (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures (default: %(default)s).",
    )
    return parser.parse_args()


def make_plots(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []

    event_df = (
        df.sort_values("start_utc", na_position="last")
        .drop_duplicates(subset=["id"])
        .copy()
    )
    event_df["center_lat"] = (event_df["llcrnrlat"] + event_df["urcrnrlat"]) / 2.0
    event_df["center_lon"] = (event_df["llcrnrlon"] + event_df["urcrnrlon"]) / 2.0

    timed_event_df = event_df.dropna(subset=["start_utc"]).copy()
    event_types = sorted(event_df["event_type"].dropna().unique())
    event_palette = build_event_palette(event_types)
    event_markers = build_event_markers(event_types)

    fig, ax = plt.subplots(figsize=(10, 5))
    img_counts = df["img_type"].value_counts().sort_values(ascending=False)
    sns.barplot(x=img_counts.index, y=img_counts.values, ax=ax, color=sns.color_palette("colorblind", 8)[0])
    ax.set_title("Image Type Frequency (All Rows)")
    ax.set_xlabel("Image Type")
    ax.set_ylabel("Count")
    saved_paths.append(save_figure(fig, output_dir, "events_image_type_frequency.png"))

    fig, ax = plt.subplots(figsize=(12, 6))
    event_counts = (
        event_df["event_type"]
        .value_counts()
        .rename_axis("event_type")
        .reset_index(name="count")
    )
    sns.barplot(
        data=event_counts,
        x="count",
        y="event_type",
        hue="event_type",
        palette=event_palette,
        ax=ax,
        legend=False,
    )
    ax.set_title("Event Type Frequency (Unique Events)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Event Type")
    saved_paths.append(save_figure(fig, output_dir, "events_event_type_frequency.png"))

    fig, ax = plt.subplots(figsize=(14, 7))
    mix = pd.crosstab(df["event_type"], df["img_type"]).sort_index()
    mix.plot(kind="bar", stacked=True, ax=ax, cmap="cividis")
    ax.set_title("Image Type Composition by Event Type")
    ax.set_xlabel("Event Type")
    ax.set_ylabel("Row Count")
    ax.legend(title="Image Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    saved_paths.append(save_figure(fig, output_dir, "events_image_type_by_event_type.png"))

    if not timed_event_df.empty:
        fig, ax = plt.subplots(figsize=(14, 5))
        daily_counts = timed_event_df.set_index("start_utc").resample("D").size()
        daily_counts.plot(ax=ax, color=sns.color_palette("colorblind", 8)[1], linewidth=1.8)
        ax.set_title("Daily Event Count (Unique Events)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Events")
        saved_paths.append(save_figure(fig, output_dir, "events_daily_timeseries.png"))

        daily_type_counts = (
            timed_event_df.set_index("start_utc")
            .groupby("event_type")
            .resample("D")
            .size()
            .rename("count")
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.lineplot(
            data=daily_type_counts,
            x="start_utc",
            y="count",
            hue="event_type",
            palette=event_palette,
            linewidth=1.5,
            ax=ax,
        )
        ax.set_title("Daily Event Count by Event Type")
        ax.set_xlabel("Date")
        ax.set_ylabel("Events per Day")
        ax.legend(title="Event Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        saved_paths.append(save_figure(fig, output_dir, "events_daily_timeseries_by_event_type.png"))

        fig, ax = plt.subplots(figsize=(12, 5))
        hourly_counts = (
            timed_event_df["start_utc"]
            .dt.hour
            .value_counts()
            .reindex(range(24), fill_value=0)
            .sort_index()
        )
        sns.barplot(x=hourly_counts.index, y=hourly_counts.values, ax=ax, color=sns.color_palette("colorblind", 8)[2])
        ax.set_title("Event Count by Hour of Day (UTC)")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Events")
        saved_paths.append(save_figure(fig, output_dir, "events_hourly_distribution.png"))

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        monthly_counts = (
            timed_event_df.assign(month=timed_event_df["start_utc"].dt.strftime("%b"))
            .groupby(["month", "event_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(month_order, fill_value=0)
        )
        monthly_props = monthly_counts.div(monthly_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

        fig, ax = plt.subplots(figsize=(14, 6))
        bottom = np.zeros(len(monthly_props), dtype=float)
        for event_type in event_types:
            values = monthly_props[event_type].to_numpy() if event_type in monthly_props.columns else np.zeros(len(monthly_props))
            ax.bar(monthly_props.index, values, bottom=bottom, label=event_type, color=event_palette[event_type])
            bottom += values
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title("Monthly Event-Type Composition (100% Stacked)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Share of Monthly Events")
        ax.legend(title="Event Type", bbox_to_anchor=(1.02, 1), loc="upper left")
        saved_paths.append(save_figure(fig, output_dir, "events_monthly_event_type_proportions.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    missing_by_img = (
        df.assign(start_missing=df["start_utc"].isna())
        .groupby("img_type")["start_missing"]
        .sum()
        .sort_values(ascending=False)
    )
    sns.barplot(x=missing_by_img.index, y=missing_by_img.values, ax=ax, color=sns.color_palette("colorblind", 8)[3])
    ax.set_title("Missing start_utc Values by Image Type")
    ax.set_xlabel("Image Type")
    ax.set_ylabel("Missing Count")
    saved_paths.append(save_figure(fig, output_dir, "events_missing_start_utc_by_image_type.png"))

    fig, ax = plt.subplots(figsize=(10, 8))
    hm = ax.hist2d(event_df["center_lon"], event_df["center_lat"], bins=[42, 28], cmap="cividis")
    cb = fig.colorbar(hm[3], ax=ax)
    cb.set_label("Event Count")
    ax.set_title("Geospatial Event Density Heatmap (All Events)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    saved_paths.append(save_figure(fig, output_dir, "events_geospatial_heatmap_all.png"))

    n_cols = 4
    n_rows = int(np.ceil(len(event_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    for idx, event_type in enumerate(event_types):
        ax = axes[idx // n_cols][idx % n_cols]
        type_df = event_df[event_df["event_type"] == event_type]
        cmap = sns.light_palette(event_palette[event_type], as_cmap=True)
        hm = ax.hist2d(type_df["center_lon"], type_df["center_lat"], bins=[30, 20], cmap=cmap)
        fig.colorbar(hm[3], ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(event_type)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    for idx in range(len(event_types), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    fig.suptitle("Geospatial Event Density by Event Type", y=1.02)
    saved_paths.append(save_figure(fig, output_dir, "events_geospatial_heatmap_by_type.png"))

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=event_df,
        x="center_lon",
        y="center_lat",
        hue="event_type",
        style="event_type",
        palette=event_palette,
        markers=event_markers,
        alpha=0.8,
        s=45,
        ax=ax,
    )
    ax.set_title("Event Center Coordinates")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Event Type", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    saved_paths.append(save_figure(fig, output_dir, "events_geospatial_centers.png"))

    if not timed_event_df.empty:
        hourly_type = (
            timed_event_df.assign(hour=timed_event_df["start_utc"].dt.hour)
            .groupby(["event_type", "hour"])
            .size()
            .rename("count")
            .reset_index()
        )

        n_cols = 4
        n_rows = int(np.ceil(len(event_types) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), squeeze=False)
        for idx, event_type in enumerate(event_types):
            ax = axes[idx // n_cols][idx % n_cols]
            series = (
                hourly_type[hourly_type["event_type"] == event_type]
                .set_index("hour")["count"]
                .reindex(range(24), fill_value=0)
            )
            ax.bar(series.index, series.values, color=event_palette[event_type], width=0.85)
            ax.set_title(event_type)
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Count")
            ax.set_xticks([0, 6, 12, 18, 23])
        for idx in range(len(event_types), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].axis("off")
        fig.suptitle("Hourly Distribution by Event Type", y=1.02)
        saved_paths.append(save_figure(fig, output_dir, "events_hourly_distribution_by_type.png"))

        months = list(range(1, 13))
        regime_hour = (
            timed_event_df.assign(
                hour=timed_event_df["start_utc"].dt.hour,
                month=timed_event_df["start_utc"].dt.month,
            )
            .groupby(["month", "hour"])
            .size()
            .unstack(fill_value=0)
            .reindex(months)
        )

        fig, ax = plt.subplots(figsize=(14, 4.5))
        sns.heatmap(
            regime_hour,
            ax=ax,
            cmap="viridis",
            linewidths=0.3,
            linecolor="white",
            cbar_kws={"label": "Event Count"},
        )
        ax.set_title("Hourly Event Regimes Across Time of Year")
        ax.set_xlabel("Hour of Day (UTC)")
        ax.set_ylabel("Month")
        saved_paths.append(save_figure(fig, output_dir, "events_month_hour_heatmap.png"))

    return saved_paths


def main() -> None:
	args = parse_args()
	events_csv = args.events_csv.resolve()
	output_dir = args.output_dir.resolve()

	if not events_csv.exists():
		raise FileNotFoundError(f"Events CSV not found: {events_csv}")

	set_default_theme()

	df = pd.read_csv(events_csv)
	df["start_utc"] = pd.to_datetime(df["start_utc"], errors="coerce", utc=True)

	row_count = len(df)
	unique_event_count = df["id"].nunique()
	missing_start_count = df["start_utc"].isna().sum()

	print(f"Loaded rows: {row_count}")
	print(f"Unique events: {unique_event_count}")
	print(f"Missing start_utc values: {missing_start_count}")

	saved_paths = make_plots(df, output_dir)

	print("Saved figures:")
	for path in saved_paths:
		print(f"- {path}")


if __name__ == "__main__":
	main()
