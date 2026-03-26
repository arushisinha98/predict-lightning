"""Visualization helpers for train.h5 EDA."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from eda_utils import save_figure


def plot_channel_presence(schema_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = schema_df.sort_values("presence_ratio", ascending=False)
    sns.barplot(data=plot_df, x="channel", y="presence_ratio", hue="channel", legend=False, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Channel Availability Across Scanned Events")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Presence Ratio")
    return save_figure(fig, output_dir, "h5_channel_presence.png")


def plot_shape_consistency(schema_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_df = schema_df.copy()
    plot_df["shape_variants"] = plot_df["shape_variants"].astype(float)
    sns.barplot(data=plot_df, x="channel", y="shape_variants", hue="channel", legend=False, ax=ax)
    ax.set_title("Shape Variants per Channel")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Unique Shape Count")
    return save_figure(fig, output_dir, "h5_shape_variants.png")


def plot_channel_percentiles(stats_df: pd.DataFrame, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = stats_df.sort_values("channel")
    x = range(len(ordered))
    ax.plot(x, ordered["p01"], marker="o", label="p01")
    ax.plot(x, ordered["p50"], marker="o", label="p50")
    ax.plot(x, ordered["p99"], marker="o", label="p99")
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(ordered["channel"]))
    ax.set_title("Sampled Channel Percentiles")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "h5_channel_percentiles.png")


def plot_lightning_hist(lightning_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if lightning_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.histplot(lightning_df["lightning_points"], bins=40, ax=ax)
    ax.set_title("Lightning Points per Event")
    ax.set_xlabel("Number of Lightning Points")
    ax.set_ylabel("Event Count")
    return save_figure(fig, output_dir, "h5_lightning_points_hist.png")


def plot_event_type_distribution(event_type_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if event_type_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = event_type_df.sort_values("count", ascending=False)
    sns.barplot(data=ordered, x="count", y="event_type", hue="event_type", legend=False, ax=ax)
    ax.set_title("Event Type Distribution (Unique Event IDs)")
    ax.set_xlabel("Count")
    ax.set_ylabel("Event Type")
    return save_figure(fig, output_dir, "h5_event_type_distribution.png")


def plot_frame_intensity_trends(frame_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if frame_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(11, 5.5))
    channels = sorted(frame_df["channel"].unique())
    palette = sns.color_palette("colorblind", n_colors=max(len(channels), 4))

    for idx, channel in enumerate(channels):
        subset = frame_df[frame_df["channel"] == channel].sort_values("frame")
        color = palette[idx]
        ax.plot(
            subset["frame"],
            subset["mean_intensity"],
            label=f"{channel} mean",
            color=color,
            linewidth=1.8,
        )
        ax.fill_between(
            subset["frame"],
            subset["p10_intensity"],
            subset["p90_intensity"],
            color=color,
            alpha=0.12,
        )

    ax.set_title("Per-Frame Intensity Trends Across 36 Timesteps")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Intensity")
    ax.legend(loc="best", ncol=2)
    return save_figure(fig, output_dir, "h5_frame_intensity_trends.png")


def plot_lightning_frame_trend(lightning_frame_df: pd.DataFrame, output_dir: Path) -> Path | None:
    if lightning_frame_df.empty:
        return None

    ordered = lightning_frame_df.sort_values("frame")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ordered["frame"], ordered["mean_lightning_count"], marker="o", label="mean")
    ax.plot(ordered["frame"], ordered["p90_lightning_count"], linestyle="--", label="p90")
    ax.set_title("Per-Frame Lightning Count Trend")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Lightning Count")
    ax.legend(loc="best")
    return save_figure(fig, output_dir, "h5_lightning_frame_trend.png")
