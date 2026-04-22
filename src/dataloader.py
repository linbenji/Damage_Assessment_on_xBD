"""
dataloader.py

XBD Dataset Dataloader
"""

from torch.utils.data import DataLoader, WeightedRandomSampler
from src.augment import XBDDatasetAugmented
import numpy as np


def get_loaders(train_files, val_files, test_files,
    img_dir, mask_dir,
    batch_size=4, num_workers=4, pin_memory = True, siamese=False, augment_train=False):
    """
    Used to load augmented XBD dataset
    """

    train_dataset = XBDDatasetAugmented(train_files, img_dir, mask_dir, siamese=siamese, augment=augment_train)
    # Note: don't augment val and test
    val_dataset = XBDDatasetAugmented(val_files, img_dir, mask_dir, siamese=siamese, augment=False)
    test_dataset = XBDDatasetAugmented(test_files, img_dir, mask_dir, siamese=siamese, augment=False)

    # Oversampling
    weights = np.array(train_dataset.sample_weights)
    weights = weights / (weights.mean() + 1e-8)
    weights = (weights + 1e-6) # normalize
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              shuffle=False, num_workers = num_workers, pin_memory = pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers = num_workers, pin_memory = pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers = num_workers, pin_memory = pin_memory)

    return train_loader, val_loader, test_loader