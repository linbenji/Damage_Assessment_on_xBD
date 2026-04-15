"""
dataloadert.py

XBD Dataset Dataloader
"""

from torch.utils.data import DataLoader
from src.augment import XBDDatasetAugmented


def get_loaders(train_files, val_files, test_files,
    img_dir, mask_dir,
    batch_size=4, num_workers=0, augment_train=False):


    train_dataset = XBDDatasetAugmented(train_files, img_dir, mask_dir, augment=augment_train)
    val_dataset = XBDDatasetAugmented(val_files, img_dir, mask_dir, augment=augment_train)
    test_dataset = XBDDatasetAugmented(test_files, img_dir, mask_dir, augment=augment_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers = num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers = num_workers)

    return train_loader, val_loader, test_loader