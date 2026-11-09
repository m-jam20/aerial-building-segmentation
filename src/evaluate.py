"""Score a checkpoint on the full-size test images, overall and per city."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.dataset import eval_transform
from src.metrics import BinaryMetrics
from src.model import build_model, normalization_for


def eval_sets(cfg):
    """Map each test set name to its (cities, image ids)."""
    return {"test": (cfg["data"]["cities"], cfg["data"]["splits"]["test"])}


def window_starts(size, window, stride):
    """Window positions along one axis, always reaching the far edge."""
    last = size - window
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.no_grad()
def predict_image(model, image, window, stride, batch_size, device):
    """Return HxW logits for a CxHxW image, averaging overlapping windows."""
    model.eval()
    _, height, width = image.shape
    total = torch.zeros(height, width)
    count = torch.zeros(height, width)
    positions = [
        (top, left)
        for top in window_starts(height, window, stride)
        for left in window_starts(width, window, stride)
    ]
    for i in range(0, len(positions), batch_size):
        batch_positions = positions[i : i + batch_size]
        batch = torch.stack([image[:, t : t + window, l : l + window] for t, l in batch_positions])
        logits = model(batch.to(device)).cpu()
        for output, (top, left) in zip(logits, batch_positions):
            total[top : top + window, left : left + window] += output[0]
            count[top : top + window, left : left + window] += 1
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="runs/baseline/best.pt")
    parser.add_argument("--predictions", default="runs/baseline/predictions")
    parser.add_argument("--set", default="test", help="name of an evaluation set in the config")
    parser.add_argument("--out", help="Defaults to results/metrics.json for the test set")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data, eval_cfg = cfg["data"], cfg["evaluate"]
    sets = eval_sets(cfg)
    if args.set not in sets:
        parser.error(f"--set must be one of {', '.join(sets)}")
    cities, ids = sets[args.set]
    out = Path(args.out or "results/metrics.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])

    predictions = Path(args.predictions)
    predictions.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(data["raw_dir"]) / "data" / "train"
    transform = eval_transform(*normalization_for(state["config"]["model"]))
    threshold = eval_cfg["threshold"]

    overall = BinaryMetrics(threshold)
    per_city = {city: BinaryMetrics(threshold) for city in cities}
    per_image = {}
    names = [(city, f"{city}{i}") for city in cities for i in ids]
    for city, name in tqdm(names):
        image = np.array(Image.open(raw_dir / "images" / f"{name}.tif").convert("RGB"))
        mask = np.array(Image.open(raw_dir / "gt" / f"{name}.tif").convert("L")) > 0
        logits = predict_image(
            model, transform(image=image)["image"],
            eval_cfg["window"], eval_cfg["stride"], eval_cfg["batch_size"], device,
        )
        targets = torch.from_numpy(mask)

        single = BinaryMetrics(threshold)
        for metrics in (overall, per_city[city], single):
            metrics.update(logits, targets)
        per_image[name] = single.compute()

        pred_mask = (torch.sigmoid(logits) > threshold).numpy().astype(np.uint8) * 255
        Image.fromarray(pred_mask).save(predictions / f"{name}.png")

    results = {
        "set": args.set,
        "checkpoint_epoch": state["epoch"],
        "overall": overall.compute(),
        "per_city": {city: metrics.compute() for city, metrics in per_city.items()},
        "per_image": per_image,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: results[key] for key in ("overall", "per_city")}, indent=2))


if __name__ == "__main__":
    main()
