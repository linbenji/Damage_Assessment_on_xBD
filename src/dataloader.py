"""
dataloadert.py

XBD Dataset Dataloader
"""

from torch.utils.data import DataLoader
from src.augment import XBDDatasetAugmented


def get_loaders(train_files, val_files, test_files,
    img_dir, mask_dir,
    batch_size=4, num_workers=4, pin_memory = True, augment_train=False):


    train_dataset = XBDDatasetAugmented(train_files, img_dir, mask_dir, augment=augment_train)
    val_dataset = XBDDatasetAugmented(val_files, img_dir, mask_dir, augment=augment_train)
    test_dataset = XBDDatasetAugmented(test_files, img_dir, mask_dir, augment=augment_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers = num_workers, pin_memory = pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers = num_workers, pin_memory = pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers = num_workers, pin_memory = pin_memory)

    return train_loader, val_loader, test_loader