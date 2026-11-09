"""Loss functions. All of them take raw logits."""
import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(logits, targets, smooth=1.0):
    """Soft Dice loss, computed per sample and then averaged."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    total = probs.sum(dim=1) + targets.sum(dim=1)
    return (1 - (2 * intersection + smooth) / (total + smooth)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice_loss(logits, targets)
