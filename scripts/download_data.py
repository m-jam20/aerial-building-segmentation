"""Download the configured subset of the Inria dataset."""
import argparse
from pathlib import Path

import numpy as np
import yaml
from huggingface_hub import snapshot_download
from PIL import Image


def subset_paths(cfg):
    data = cfg["data"]
    groups = [(data["cities"], [i for split in data["splits"].values() for i in split])]
    return [
        f"data/train/{folder}/{city}{i}.tif"
        for cities, ids in groups
        for city in cities
        for i in ids
        for folder in ("images", "gt")
    ]


def verify(paths, raw_dir):
    for path in paths:
        with Image.open(raw_dir / path) as img:
            if img.size != (5000, 5000):
                raise ValueError(f"{path}: unexpected size {img.size}")
            if "/gt/" in path and not set(np.unique(img)) <= {0, 255}:
                raise ValueError(f"{path}: mask is not binary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    paths = subset_paths(cfg)
    data = cfg["data"]
    raw_dir = Path(data["raw_dir"])
    snapshot_download(
        repo_id=data["repo_id"],
        repo_type="dataset",
        revision=data["revision"],
        allow_patterns=paths,
        local_dir=raw_dir,
    )
    verify(paths, raw_dir)
    print(f"Downloaded and verified {len(paths)} files in {raw_dir}")


if __name__ == "__main__":
    main()
