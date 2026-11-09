"""Train the U-Net and keep the checkpoint with the best validation IoU."""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from src.dataset import TileDataset, train_transform, val_transform
from src.losses import BCEDiceLoss
from src.metrics import BinaryMetrics
from src.model import ARCHITECTURES, build_model, normalization_for

HISTORY_FIELDS = [
    "epoch", "lr", "train_loss", "val_loss", "val_iou", "val_dice",
    "val_precision", "val_recall", "seconds",
]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_worker(_worker_id):
    """Give each worker its own augmentation stream."""
    get_worker_info().dataset.transform.set_random_seed(torch.initial_seed() % 2**32)


def make_loaders(cfg):
    tiles = Path(cfg["data"]["tiles_dir"])
    train_cfg, val_cfg = cfg["train"], cfg["val"]
    workers = train_cfg["num_workers"]
    generator = torch.Generator().manual_seed(cfg["seed"])
    mean, std = normalization_for(cfg["model"])
    train_set = TileDataset(
        tiles / "train", train_transform(train_cfg["crop_size"], cfg["seed"], mean, std)
    )
    val_set = TileDataset(tiles / "val", val_transform(val_cfg["crop_size"], mean, std))
    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(val_set, batch_size=val_cfg["batch_size"], num_workers=workers, pin_memory=True)
    return train_loader, val_loader


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(images), masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()}, stopping")
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    metrics = BinaryMetrics()
    for images, masks in tqdm(loader, desc="val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += loss_fn(logits, masks).item()
        metrics.update(logits, masks)
    return total / len(loader), metrics.compute()


def fit(cfg, out_dir, device, resume=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])

    train_loader, val_loader = make_loaders(cfg)
    model = build_model(cfg["model"]).to(device)
    loss_fn = BCEDiceLoss(**cfg["loss"])
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])

    start_epoch, best_iou = 1, -1.0
    if resume:
        state = torch.load(out_dir / "last.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_iou = state["epoch"] + 1, state["best_iou"]

    history_path = out_dir / "history.csv"
    if not resume:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, HISTORY_FIELDS).writeheader()

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        started = time.time()
        lr = scheduler.get_last_lr()[0]
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{name}": value for name, value in val.items()},
            "seconds": time.time() - started,
        }
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, HISTORY_FIELDS).writerow(row)
        print(
            f"epoch {epoch}/{train_cfg['epochs']} train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
            f"val_iou {val['iou']:.4f} val_dice {val['dice']:.4f} ({row['seconds']:.0f}s)",
            flush=True,
        )

        if val["iou"] > best_iou:
            best_iou = val["iou"]
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch, "val": val}, out_dir / "best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou,
            },
            out_dir / "last.pt",
        )
    return best_iou


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default="runs/baseline")
    parser.add_argument("--epochs", type=int, help="Override the number of epochs")
    parser.add_argument(
        "--arch",
        choices=ARCHITECTURES,
        help="Override model.arch. Everything else stays as configured, which "
        "is what keeps a benchmark between architectures fair.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt in --out")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.arch:
        cfg["model"]["arch"] = args.arch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_iou = fit(cfg, args.out, device, args.resume)
    print(f"Best validation IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
