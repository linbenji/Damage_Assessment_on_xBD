"""
train.py

Evaluation Utilities for XBD Dataset
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

from eval import ConfusionMatrixTracker, validate, CLASS_NAMES

# Class TverskyLoss provided by LLM
class TverskyLoss(nn.Module):
    """
    Tversky loss for multi-class segmentation.

    alpha > beta  →  penalizes false negatives more (boosts recall)
    alpha < beta  →  penalizes false positives more (boosts precision)
    alpha = beta = 0.5  →  equivalent to Dice loss
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6, num_classes=5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # sum over batch, H, W
        TP = (probs * one_hot).sum(dim=dims)
        FP = (probs * (1 - one_hot)).sum(dim=dims)
        FN = ((1 - probs) * one_hot).sum(dim=dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1.0 - tversky.mean()

# Class ComboLoss created w/ assistance by LLM
class ComboLoss(nn.Module):
    """
    Weighted combination: ce_weight * CrossEntropy + tversky_weight * TverskyLoss
    """

    def __init__(self, class_weights, ce_weight=0.5, tversky_weight=0.5,
                 tversky_alpha=0.7, tversky_beta=0.3, num_classes=5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta,
                                   num_classes=num_classes)
        self.ce_weight = ce_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        return (self.ce_weight * self.ce(logits, targets) +
                self.tversky_weight * self.tversky(logits, targets))

# Modified Version From :medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -np.inf if mode == 'max' else np.inf
        self.counter = 0
        self.stop_training = False

    def step(self, value):
        """Check if training should stop. Returns True if improved."""
        if self.mode == 'max':
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop_training = True

        return improved

def train_one_epoch(model, loader, criterion, optimizer, device, num_classes=5, siamese=False):
    model.train()
    running_loss = 0.0
    tracker = ConfusionMatrixTracker(num_classes)

    for batch in tqdm(loader, desc="Train", leave=False):

        optimizer.zero_grad()

        if siamese:
            pre, post, masks = batch
            pre = pre.to(device)
            post = post.to(device)
            masks = masks.to(device)

            outputs = model(pre, post)

        else:
            images, masks = batch
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * masks.size(0)
        tracker.update(outputs.argmax(dim=1), masks)

    return running_loss / len(loader.dataset), tracker.compute()



# Full training loop created w/ assistance by LLM
def run_training(model, train_loader, val_loader,
        criterion, optimizer, scheduler, device,
        num_epochs=25, patience=5, num_classes=5,
        save_path="best_model.pth", verbose=True,
        siamese=False):

    early_stop = EarlyStopping(patience=patience, mode='max')
    history = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": [], "lr": []}

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes, siamese=siamese
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, num_classes, siamese=siamese
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_metrics["miou"])
        history["val_miou"].append(val_metrics["miou"])

        if verbose:
            print(f"\nEpoch {epoch}/{num_epochs}  (lr={current_lr:.2e})")
            print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
            print(f"  Train mIoU: {train_metrics['miou']:.4f}  |  Val mIoU: {val_metrics['miou']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']:.4f}  |  Val Acc:  {val_metrics['accuracy']:.4f}")

            print(f"  Val Per-Class:")
            print(f"    {'Class':15s} {'IoU':>8s} {'Prec':>8s} {'Recall':>8s}")
            print(f"    {'-' * 41}")
            for i, name in enumerate(CLASS_NAMES):
                print(f"    {name:15s} "
                      f"{val_metrics['per_class_iou'][i]:8.4f} "
                      f"{val_metrics['per_class_precision'][i]:8.4f} "
                      f"{val_metrics['per_class_recall'][i]:8.4f}")

        # Save best + early stopping
        is_best = early_stop.step(val_metrics["miou"])
        if is_best:
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"  *** Saved new best model (mIoU={val_metrics['miou']:.4f}) ***")

        if early_stop.stop_training:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    return history


def plot_train_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_miou"], marker="o", label="Train")
    axes[1].plot(epochs, history["val_miou"], marker="o", label="Val")
    axes[1].set_title("Mean IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], marker="o", label="LR")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log")
    axes[2].legend()

    plt.tight_layout()
    plt.show()