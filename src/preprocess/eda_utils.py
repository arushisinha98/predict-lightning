"""Shared plotting helpers for preprocessing and EDA scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_figure(fig: plt.Figure, output_dir: Path, filename: str, dpi: int = 150) -> Path:
    """Persist and close a figure, returning the output path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_event_palette(event_types: list[str]) -> dict[str, tuple[float, float, float]]:
    """Build a deterministic colorblind-safe palette keyed by event type."""
    colors = sns.color_palette("colorblind", n_colors=max(len(event_types), 8))
    return {event_type: colors[idx] for idx, event_type in enumerate(event_types)}


def build_event_markers(event_types: list[str]) -> dict[str, str]:
    """Assign deterministic markers keyed by event type."""
    marker_cycle = ["o", "s", "D", "^", "v", "P", "X", "*"]
    return {
        event_type: marker_cycle[idx % len(marker_cycle)]
        for idx, event_type in enumerate(event_types)
    }


def set_default_theme() -> None:
    """Apply a repository-wide plotting theme for consistency."""
    sns.set_theme(style="whitegrid")
