"""
augment.py

XBD Dataset with Geometric Augmentations

"""

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image

# class XBDDatasetAugmented created w/ Assistance from LLM
class XBDDatasetAugmented(Dataset):
    def __init__(self, file_list, img_dir, mask_dir, augment=False, siamese=False):
        self.file_list = file_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.augment = augment
        self.siamese = siamese

        # Define sample weights for oversampling
        self.sample_weights = []
        for fname in self.file_list:
            mask_path = os.path.join(self.mask_dir, fname)
            mask = np.array(Image.open(mask_path)).astype(np.int64)
            w = self.get_sample_weight(mask)
            self.sample_weights.append(w)

    def __len__(self):
        return len(self.file_list)

    def _apply_augmentations(self, pre, post, mask):

        # Random horizontal flip
        if np.random.rand() < 0.5:
            pre = np.flip(pre, axis=2).copy()
            post = np.flip(post, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Random vertical flip
        if np.random.rand() < 0.5:
            pre = np.flip(pre, axis=1).copy()
            post = np.flip(post, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Random 90° rotation (k = 0, 1, 2, or 3 times)
        k = np.random.randint(0, 4)
        if k > 0:
            pre = np.rot90(pre, k, axes=(1, 2)).copy()
            post = np.rot90(post, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return pre, post, mask


    def __getitem__(self, idx):
        post_name = self.file_list[idx]
        pre_name = post_name.replace("post_disaster", "pre_disaster")

        pre = np.array(Image.open(os.path.join(self.img_dir, pre_name))).astype(np.float32) / 255.0
        post = np.array(Image.open(os.path.join(self.img_dir, post_name))).astype(np.float32) / 255.0
        mask = np.array(Image.open(os.path.join(self.mask_dir, post_name))).astype(np.int64)

        pre = np.transpose(pre, (2, 0, 1))
        post = np.transpose(post, (2, 0, 1))

        if self.augment:
            pre, post, mask = self._apply_augmentations(pre, post, mask)

        # Normalize augmented pre and post datasets to [-1, 1]
        pre = (pre - 0.5) / 0.5
        post = (post - 0.5) / 0.5


        if self.siamese: # do not concatenate pre post and mask if siamese
            return (
                torch.tensor(pre, dtype=torch.float32),
                torch.tensor(post, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.long),
            )
        else: # Add normalized difference if not Siamese (used for change detection, in range [-1,1])
            diff = (post - pre) / 2.0
            image = np.concatenate([pre, post, diff], axis=0)

            return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.long)

    def get_sample_weight(self, mask):
        """
        Gets weights for major/destroyed classes for oversampling.
        Uses tile-level weights (ie, not pixel density) to focus
        on presence of major and destroyed classes.
        """
        if np.any(mask == 4):
            return 15.0
        if np.any(mask == 3):
            return 10.0
        if np.any(mask == 2):
            return 3.0
        else:
            return 1.0
