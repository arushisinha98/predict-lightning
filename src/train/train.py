"""Configurable training entrypoint for lightning prediction models."""

from __future__ import annotations

import argparse
import difflib
import random
from pathlib import Path

import numpy as np
import torch

from data import LightningFrameDataset, create_dataloader, list_event_ids
from losses import build_loss, list_losses
from models import build_model, list_models
from trainer import fit


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightning prediction models.")
    parser.add_argument("--data-path", type=Path, default=REPO_ROOT / "data" / "preprocessed_train.h5")
    parser.add_argument("--model", choices=list_models(), default="unet")
    parser.add_argument("--loss", choices=list_losses(), default="multitask")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "outputs" / "checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> str:
    if requested_device.startswith("cuda") and torch.cuda.is_available():
        return requested_device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
    return "cpu"


def split_event_ids(event_ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    ids = event_ids.copy()
    rng.shuffle(ids)

    val_size = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:val_size]
    train_ids = ids[val_size:]
    if not train_ids:
        raise ValueError("Train split is empty. Reduce val_ratio or increase events.")
    return train_ids, val_ids


def validate_data_path(data_path: Path) -> Path:
    """Validate HDF5 path and provide typo-aware guidance."""
    if data_path.exists():
        return data_path

    data_dir = REPO_ROOT / "data"
    known_h5 = sorted([p.name for p in data_dir.glob("*.h5")]) if data_dir.exists() else []
    suggestion = None

    if known_h5:
        matches = difflib.get_close_matches(data_path.name, known_h5, n=1, cutoff=0.55)
        if matches:
            suggestion = data_dir / matches[0]
        elif "preprocess_train.h5" in data_path.name and "preprocessed_train.h5" in known_h5:
            suggestion = data_dir / "preprocessed_train.h5"

    message = [f"Data file not found: {data_path}"]
    if suggestion is not None:
        message.append(f"Did you mean: {suggestion} ?")
    if known_h5:
        message.append("Available .h5 files in data/: " + ", ".join(known_h5))

    raise FileNotFoundError("\n".join(message))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    cwd = Path.cwd()
    data_path = args.data_path if args.data_path.is_absolute() else cwd / args.data_path
    data_path = validate_data_path(data_path)
    event_ids = list_event_ids(data_path)
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    train_ids, val_ids = split_event_ids(event_ids, args.val_ratio, args.seed)

    train_dataset = LightningFrameDataset(train_ids, file_path=data_path, test_data=False)
    val_dataset = LightningFrameDataset(val_ids, file_path=data_path, test_data=False)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, device=device).to(device)
    loss_fn = build_loss(args.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_dir = (
        args.checkpoint_dir if args.checkpoint_dir.is_absolute() else cwd / args.checkpoint_dir
    )

    print(
        f"Training config: model={args.model} loss={args.loss} "
        f"events={len(event_ids)} train_events={len(train_ids)} val_events={len(val_ids)} device={device}"
    )

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
