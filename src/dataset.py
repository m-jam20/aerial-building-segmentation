"""Dataset and augmentations for the image and mask tiles."""
from pathlib import Path

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset


# The from-scratch U-Net was trained with these. Pretrained encoders need
# ImageNet statistics instead; src.model.normalization_for gives the right
# pair for an architecture, and callers pass it in.
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


def _finalize(mean=DEFAULT_MEAN, std=DEFAULT_STD):
    return [A.Normalize(mean=mean, std=std), ToTensorV2()]


def train_transform(crop_size, seed=None, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    return A.Compose(
        [
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            *_finalize(mean, std),
        ],
        seed=seed,
    )


def val_transform(crop_size, mean=DEFAULT_MEAN, std=DEFAULT_STD):
    return A.Compose([A.CenterCrop(crop_size, crop_size), *_finalize(mean, std)])


def eval_transform(mean=DEFAULT_MEAN, std=DEFAULT_STD):
    """Normalization only, for scoring full images."""
    return A.Compose(_finalize(mean, std))


class TileDataset(Dataset):
    """Returns (image, mask): a 3xHxW float image and a 1xHxW float 0/1 mask."""

    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.names = sorted(p.name for p in (self.root / "images").glob("*.png"))
        if not self.names:
            raise FileNotFoundError(f"No tiles found in {self.root / 'images'}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = np.array(Image.open(self.root / "images" / name).convert("RGB"))
        mask = (np.array(Image.open(self.root / "masks" / name).convert("L")) > 0).astype(np.uint8)
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()
