"""Pixel metrics for binary segmentation."""
import torch


def _ratio(numerator, denominator, undefined):
    return numerator / denominator if denominator else undefined


class BinaryMetrics:
    """Accumulates pixel counts, so scores are over all pixels, not a mean of batches.

    IoU and Dice are 1 when both prediction and target are empty. Precision and
    recall are 0 when undefined.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.tp = self.fp = self.fn = 0

    @torch.no_grad()
    def update(self, logits, targets):
        preds = torch.sigmoid(logits) > self.threshold
        targets = targets.bool()
        self.tp += (preds & targets).sum().item()
        self.fp += (preds & ~targets).sum().item()
        self.fn += (~preds & targets).sum().item()

    def compute(self):
        tp, fp, fn = self.tp, self.fp, self.fn
        return {
            "iou": _ratio(tp, tp + fp + fn, 1.0),
            "dice": _ratio(2 * tp, 2 * tp + fp + fn, 1.0),
            "precision": _ratio(tp, tp + fp, 0.0),
            "recall": _ratio(tp, tp + fn, 0.0),
        }
