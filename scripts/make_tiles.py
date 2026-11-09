"""Cut the downloaded images and masks into non-overlapping tiles."""
import argparse
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


def tile_pair(image, mask, name, out_dir, tile_size, keep_empty, rng):
    kept = 0
    for top in range(0, image.shape[0], tile_size):
        for left in range(0, image.shape[1], tile_size):
            mask_tile = mask[top : top + tile_size, left : left + tile_size]
            if not mask_tile.any() and rng.random() >= keep_empty:
                continue
            image_tile = image[top : top + tile_size, left : left + tile_size]
            tile_name = f"{name}_{top}_{left}.png"
            Image.fromarray(image_tile).save(out_dir / "images" / tile_name, compress_level=1)
            Image.fromarray(mask_tile).save(out_dir / "masks" / tile_name, compress_level=1)
            kept += 1
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data = cfg["data"]

    if data["tile_size"] <= 0 or 5000 % data["tile_size"]:
        raise ValueError("tile_size must divide 5000 exactly")

    rng = random.Random(cfg["seed"])
    raw_dir = Path(data["raw_dir"]) / "data" / "train"
    for split, ids in data["splits"].items():
        out_dir = Path(data["tiles_dir"]) / split
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        keep_empty = data["keep_empty_fraction"] if split == "train" else 1.0

        kept = 0
        names = [f"{city}{i}" for city in data["cities"] for i in ids]
        for name in tqdm(names, desc=split):
            image = np.array(Image.open(raw_dir / "images" / f"{name}.tif").convert("RGB"))
            mask = np.array(Image.open(raw_dir / "gt" / f"{name}.tif").convert("L"))
            kept += tile_pair(image, mask, name, out_dir, data["tile_size"], keep_empty, rng)
        print(f"{split}: {kept} tiles from {len(names)} images")


if __name__ == "__main__":
    main()
