"""Render qualitative panels and training curves for the README.

Panels show image / ground truth / prediction / error overlay. The overlay
colours true positives, false positives and false negatives separately,
because the two failure modes need telling apart: a tile the model left
blank and a tile it painted over are both wrong in a single-colour error
map, yet they call for opposite fixes.

Test images appear here only to report results. No threshold, setting or
checkpoint is ever chosen from them.
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image

# Run as `python scripts/make_figures.py` from the repo root, as the README
# invokes the other scripts; that puts scripts/ on the path, not the root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import eval_transform
from src.evaluate import eval_sets, predict_image
from src.metrics import BinaryMetrics
from src.model import build_model, normalization_for

# Slots 1-3 of the reference categorical palette, documented as clearing the
# all-pairs colour-vision gates in both light and dark modes.
TP_COLOUR = "#1baf7a"
FP_COLOUR = "#eb6834"
FN_COLOUR = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#d9d8d4"


def _hex_to_rgb(value):
    return np.array([int(value[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)


def shrink(array, width):
    """Downsample to `width` px for display. Nearest keeps masks crisp."""
    if array.shape[1] <= width:
        return array
    height = round(array.shape[0] * width / array.shape[1])
    return np.array(Image.fromarray(array).resize((width, height), Image.NEAREST))


def busiest_window(truth, prediction, size, block=25):
    """Top-left corner of the `size` window holding the most building activity.

    Whole Inria images are 5000 px wide, so a building is one or two pixels
    once a panel is downsampled, and a sparse rural image renders as an empty
    grey square. Cropping is a display choice over already-scored results: it
    selects no model, threshold or setting. The window counts truth and prediction together so it cannot favour one
    failure mode - a missed building and a false alarm both pull it.
    """
    height, width = truth.shape
    size = min(size, height, width)
    if size == min(height, width) and height == width:
        return 0, 0
    activity = (truth | prediction).astype(np.float32)
    # Block-mean down to a coarse grid, then slide the window over that.
    small = np.array(
        Image.fromarray(activity * 255).resize(
            (max(width // block, 1), max(height // block, 1)), Image.BOX
        ),
        dtype=np.float32,
    )
    span = max(size // block, 1)
    if small.shape[0] <= span or small.shape[1] <= span:
        return 0, 0
    integral = np.pad(small, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[span:, span:]
        - integral[:-span, span:]
        - integral[span:, :-span]
        + integral[:-span, :-span]
    )
    if sums.max() <= 0:
        return (height - size) // 2, (width - size) // 2
    row, column = np.unravel_index(int(sums.argmax()), sums.shape)
    return (
        int(min(row * block, height - size)),
        int(min(column * block, width - size)),
    )


def error_overlay(image, truth, prediction, alpha=0.55):
    """Grey base image with TP, FP and FN painted over it."""
    grey = image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    base = np.repeat((0.35 * grey + 0.45 * 255)[..., None], 3, axis=2)
    for mask, colour in (
        (truth & prediction, TP_COLOUR),
        (~truth & prediction, FP_COLOUR),
        (truth & ~prediction, FN_COLOUR),
    ):
        if mask.any():
            base[mask] = (1 - alpha) * base[mask] + alpha * _hex_to_rgb(colour)
    return base.clip(0, 255).astype(np.uint8)


def load_prediction(cache, name, shape):
    """Reuse a mask saved by src.evaluate, if it matches the image."""
    path = cache / f"{name}.png"
    if not path.exists():
        return None
    saved = np.array(Image.open(path).convert("L")) > 127
    return saved if saved.shape == shape else None


def predict(model, image, cfg, device, normalization):
    eval_cfg = cfg["evaluate"]
    logits = predict_image(
        model,
        eval_transform(*normalization)(image=image)["image"],
        eval_cfg["window"],
        eval_cfg["stride"],
        eval_cfg["batch_size"],
        device,
    )
    return (torch.sigmoid(logits) > eval_cfg["threshold"]).numpy()


def score(truth, prediction, threshold):
    metrics = BinaryMetrics(threshold)
    metrics.tp = int((truth & prediction).sum())
    metrics.fp = int((~truth & prediction).sum())
    metrics.fn = int((truth & ~prediction).sum())
    return metrics.compute()


def panels(names, cfg, checkpoint, cache, out_path, width, device, crop=0):
    """One row per image: image, ground truth, prediction, error overlay."""
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model = build_model(state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    # Each checkpoint carries its own config, so a model is always drawn with
    # the preprocessing it was trained with.
    normalization = normalization_for(state["config"]["model"])

    raw_dir = Path(cfg["data"]["raw_dir"]) / "data" / "train"
    threshold = cfg["evaluate"]["threshold"]
    rows = []
    for name in names:
        image = np.array(Image.open(raw_dir / "images" / f"{name}.tif").convert("RGB"))
        truth = np.array(Image.open(raw_dir / "gt" / f"{name}.tif").convert("L")) > 0
        prediction = load_prediction(cache, name, truth.shape)
        if prediction is None:
            print(f"{name}: no cached prediction, running inference")
            prediction = predict(model, image, cfg, device, normalization)
        else:
            print(f"{name}: using cached prediction")
        # Score the whole image, so the row label matches results/*.json, then
        # crop only what is drawn.
        scores = score(truth, prediction, threshold)
        if crop:
            top, left = busiest_window(truth, prediction, crop)
            window = (slice(top, top + crop), slice(left, left + crop))
            image, truth, prediction = image[window], truth[window], prediction[window]
        rows.append(
            (
                name,
                scores,
                [
                    shrink(image, width),
                    shrink(truth.astype(np.uint8) * 255, width),
                    shrink(prediction.astype(np.uint8) * 255, width),
                    shrink(error_overlay(image, truth, prediction), width),
                ],
            )
        )

    titles = ["Image", "Ground truth", "Prediction", "Errors"]
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(13, 3.3 * len(rows) + 1.1), squeeze=False, facecolor=SURFACE
    )
    for row, (name, scores, images) in enumerate(rows):
        for column, (axis, panel) in enumerate(zip(axes[row], images)):
            if panel.ndim == 3:
                axis.imshow(panel, interpolation="nearest")
            else:
                axis.imshow(panel, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(titles[column], color=INK, fontsize=11, pad=8)
        axes[row][0].set_ylabel(
            "{}\nIoU {:.3f}\nprec {:.3f}\nrec {:.3f}".format(
                name, scores["iou"], scores["precision"], scores["recall"]
            ),
            color=INK_SOFT,
            fontsize=9,
            rotation=0,
            ha="right",
            va="center",
            labelpad=42,
        )

    figure.legend(
        handles=[
            Patch(facecolor=TP_COLOUR, label="Correct (true positive)"),
            Patch(facecolor=FP_COLOUR, label="False alarm (false positive)"),
            Patch(facecolor=FN_COLOUR, label="Missed (false negative)"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=INK_SOFT,
        bbox_to_anchor=(0.5, 0.004),
    )
    detail = f", {crop}px crop (scores are whole-image)" if crop else ""
    figure.suptitle(
        f"Predictions at threshold {threshold}, checkpoint epoch {state['epoch']}{detail}",
        color=INK,
        fontsize=12.5,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.975))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_path}")


def curves(history_path, out_path):
    """Loss and validation IoU as separate panels: two scales, never one axis."""
    with open(history_path) as f:
        history = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not history:
        raise ValueError(f"{history_path} has no rows")
    epochs = [row["epoch"] for row in history]

    figure, (loss_axis, iou_axis) = plt.subplots(1, 2, figsize=(11, 4.1), facecolor=SURFACE)
    for axis in (loss_axis, iou_axis):
        axis.set_facecolor(SURFACE)
        axis.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_SOFT, labelsize=9)
        axis.set_xlabel("Epoch", color=INK_SOFT, fontsize=10)
        # Epochs are whole numbers; the default locator invents 2.5, 7.5, ...
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    loss_axis.plot(
        epochs, [r["train_loss"] for r in history], color=FN_COLOUR, linewidth=2, label="Train"
    )
    loss_axis.plot(
        epochs, [r["val_loss"] for r in history], color=FP_COLOUR, linewidth=2, label="Validation"
    )
    loss_axis.set_title("BCE + Dice loss", color=INK, fontsize=11.5, loc="left")
    loss_axis.legend(frameon=False, fontsize=9.5, labelcolor=INK_SOFT)

    ious = [r["val_iou"] for r in history]
    iou_axis.plot(epochs, ious, color=FN_COLOUR, linewidth=2)
    iou_axis.set_title("Validation IoU", color=INK, fontsize=11.5, loc="left")
    best = int(np.argmax(ious))
    iou_axis.plot(
        [epochs[best]],
        [ious[best]],
        marker="o",
        markersize=8,
        color=FN_COLOUR,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=5,
    )
    # Best epoch is often the last one; anchor the label inward so it cannot
    # run off the right edge.
    span = epochs[-1] - epochs[0]
    near_end = span > 0 and epochs[best] > epochs[0] + 0.72 * span
    iou_axis.annotate(
        f"best {ious[best]:.3f} (epoch {int(epochs[best])})",
        xy=(epochs[best], ious[best]),
        # The curve plateaus high near the end, so drop the label well clear
        # of it rather than the usual short offset.
        xytext=(8, -42) if near_end else (-12, -22),
        textcoords="offset points",
        color=INK_SOFT,
        fontsize=9.5,
        ha="right" if near_end else "left",
    )

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140, facecolor=SURFACE)
    plt.close(figure)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--run", default="runs/baseline", help="Holds history.csv and predictions/")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--width", type=int, default=1100, help="Display width per panel, px")
    parser.add_argument("--set", default="test", help="name of an evaluation set in the config")
    parser.add_argument("--images", nargs="+", help="Image names; overrides --set")
    parser.add_argument(
        "--crop",
        type=int,
        default=0,
        help="Draw only the busiest NxN window. Whole 5000px images hide sparse "
        "buildings; scores stay whole-image. 0 draws the full image.",
    )
    parser.add_argument("--skip-curves", action="store_true")
    parser.add_argument("--skip-panels", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run, out = Path(args.run), Path(args.out)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.skip_panels:
        if args.images:
            names, label = args.images, "selected"
        else:
            cities, ids = eval_sets(cfg)[args.set]
            names, label = [f"{city}{ids[0]}" for city in cities], args.set
        panels(
            names,
            cfg,
            args.checkpoint,
            run / "predictions",
            out / f"panels_{label}.png",
            args.width,
            device,
            args.crop,
        )

    if not args.skip_curves:
        curves(run / "history.csv", out / "curves.png")


if __name__ == "__main__":
    main()
