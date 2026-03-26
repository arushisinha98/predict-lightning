"""Download dataset files from Hugging Face into the local data directory.

Usage:
    pixi run python src/preprocess/download.py
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DATASET_REPO_ID = "benmoseley/ese-dl-2025-26-group-project"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output_dir = repo_root / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Download all files from the Hugging Face dataset repository "
            f"'{DATASET_REPO_ID}' into a local directory."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=(
            "Destination folder for downloaded data "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision/branch/tag to download (default: %(default)s).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token for gated/private datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Downloading dataset '{DATASET_REPO_ID}' (revision: {args.revision}) "
        f"to {output_dir}..."
    )

    local_path = snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(output_dir),
        token=args.token,
    )

    print(f"Download complete. Files available in: {local_path}")


if __name__ == "__main__":
    main()
