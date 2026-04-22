"""
eval.py

Contains evaluation utilities for models trained
on the xBD Dataset
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


CLASS_NAMES = ["No Damage", "Minor", "Major", "Destroyed", "Unclassified"]

# class ConfusionMatrixTracker created w/ Assistance from LLM
class ConfusionMatrixTracker:
    """
    Accumulates a global confusion matrix across all batches,
    then computes unbiased per-class IoU, precision, recall.

    Usage:
        tracker = ConfusionMatrixTracker(num_classes=5)
        for preds, targets in loader:
            tracker.update(preds, targets)
        metrics = tracker.compute()
    """

    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.matrix.fill(0)

    def update(self, preds, targets):
        """Vectorized update. preds/targets: [B, H, W] tensors."""
        preds_np = preds.cpu().numpy().flatten()
        targets_np = targets.cpu().numpy().flatten()
        indices = targets_np * self.num_classes + preds_np
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Returns dict containing the following eval metrics:
            - per_class_iou:       [C] array
            - per_class_precision:  [C] array
            - per_class_recall:     [C] array
            - miou:                 scalar
            - damage_miou:         scalar
            - accuracy:             scalar
            - macro_f1:             scalar
            - confusion_matrix:     [C, C] array
        """
        cm = self.matrix
        TP = np.diag(cm)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP

        # Per-class metrics
        denom_iou = TP + FP + FN
        iou = np.where(denom_iou > 0, TP / denom_iou, np.nan)
        denom_prec = TP + FP
        precision = np.where(denom_prec > 0, TP / denom_prec, np.nan)
        denom_rec = TP + FN
        recall = np.where(denom_rec > 0, TP / denom_rec, np.nan)
        accuracy = TP.sum() / cm.sum() if cm.sum() > 0 else 0.0

        # Mean iou
        miou = np.nanmean(iou)

        # Damage only miou (Minor, Major, Destroyed classes)
        damage_miou = np.nanmean(iou[[1, 2, 3]])

        # Macro_f1
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            np.nan
        )
        macro_f1 = np.nanmean(f1)

        return {
            "per_class_iou": iou,
            "per_class_precision": precision,
            "per_class_recall": recall,
            "miou": miou,
            "damage_miou": damage_miou,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "confusion_matrix": cm,
        }

@torch.no_grad()
def validate(model, loader, criterion, device, num_classes = 5, siamese=False):
    """
    Used to evaluate model on validation set
    """
    model.eval()
    running_loss = 0.0
    tracker = ConfusionMatrixTracker(num_classes)

    for batch in loader:
        if siamese:
            pre, post, masks = batch
            pre, post = pre.to(device), post.to(device)
            outputs = model(pre, post)
        else:
            images, masks = batch
            images = images.to(device)
            outputs = model(images)

        masks = masks.to(device)

        loss = criterion(outputs, masks)

        running_loss += loss.item() * masks.size(0)
        tracker.update(outputs.argmax(dim=1), masks)


    return running_loss / len(loader.dataset), tracker.compute()


def test_evaluation(model, test_loader, criterion, device, save_path, num_classes=5, siamese=False):
    """
    Used to evaluate a trained model using the test set.
    Prints and returns loss and segmentation metrics.
    """

    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_metrics = validate(model, test_loader, criterion, device, num_classes, siamese=siamese)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test mIoU: {test_metrics['miou']:.4f}")
    print(f"Test Acc:  {test_metrics['accuracy']:.4f}")
    print(f"\n{'Class':15s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s}")

    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:15s} "
              f"{test_metrics['per_class_iou'][i]:8.4f} "
              f"{test_metrics['per_class_precision'][i]:8.4f} "
              f"{test_metrics['per_class_recall'][i]:8.4f}")

    return test_loss, test_metrics
