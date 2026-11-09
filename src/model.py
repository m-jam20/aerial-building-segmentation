"""U-Net for binary segmentation (Ronneberger et al., 2015)."""
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Returns one channel of raw logits, so pair it with a logit-based loss."""

    def __init__(self, in_channels=3, base_channels=32, depth=4):
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2**i for i in range(depth + 1)]

        self.pool = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels[:-1]:
            self.encoders.append(DoubleConv(previous, width))
            previous = width
        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(width * 2, width, 2, stride=2))
            self.decoders.append(DoubleConv(width * 2, width))

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        factor = 2**self.depth
        if x.shape[-2] % factor or x.shape[-1] % factor:
            raise ValueError(f"Input size {tuple(x.shape[-2:])} must be divisible by {factor}")

        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upsample(x), skip], dim=1))

        return self.head(x)


# Normalization belongs to the architecture, not to taste. A pretrained encoder
# was trained on ImageNet statistics and sees wrong-looking inputs without them,
# which would make a benchmark against it meaningless. The hand-written U-Net
# learned on 0.5/0.5 and must keep it to stay reproducible.
HALF_NORMALIZATION = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
IMAGENET_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

DEFAULT_ARCH = "unet_scratch"
PRETRAINED_ARCHS = ("unet_resnet34", "deeplabv3plus_resnet34")
ARCHITECTURES = (DEFAULT_ARCH,) + PRETRAINED_ARCHS


def normalization_for(model_cfg):
    """The (mean, std) the architecture in `model_cfg` expects."""
    arch = (model_cfg or {}).get("arch", DEFAULT_ARCH)
    return HALF_NORMALIZATION if arch == DEFAULT_ARCH else IMAGENET_NORMALIZATION


def build_model(model_cfg):
    """Build the architecture named by `model_cfg["arch"]`.

    Checkpoints written before the benchmark have no "arch" key. A missing one
    means the hand-written U-Net, so those checkpoints stay loadable and keep
    their original preprocessing.
    """
    cfg = dict(model_cfg or {})
    arch = cfg.pop("arch", DEFAULT_ARCH)
    if arch == DEFAULT_ARCH:
        return UNet(**cfg)
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {', '.join(ARCHITECTURES)}")
    # base_channels and depth size the hand-written U-Net only.
    for key in ("base_channels", "depth"):
        cfg.pop(key, None)
    # Imported here because it pulls in timm and torchvision, which the
    # from-scratch path never needs.
    import segmentation_models_pytorch as smp

    builder = {"unet_resnet34": smp.Unet, "deeplabv3plus_resnet34": smp.DeepLabV3Plus}[arch]
    return builder("resnet34", encoder_weights="imagenet", in_channels=3, classes=1, **cfg)
