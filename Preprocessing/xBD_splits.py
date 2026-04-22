"""
xBD_splits.py

Used to create train/val/test splits
"""


import os
from collections import defaultdict
import random

def extract_event(filename):
    """
    Simply extracts and returns event name from filename
    """
    return filename.split("_")[0]

def create_splits(img_dir, mask_dir, seed=42):
    """
    Creates train/val/test splits for xBD in an event-aware
    manner. In other words, tiles from the same disaster images
    are grouped together.
    """


    imgs = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    # Extract Disaster by type

    # Group Tiles by Events
    event_to_files = defaultdict(list)

    for f in imgs:
        event = extract_event(f)
        event_to_files[event].append(f)

    # Split by Events (80% Train, 10% Val, 10% Test)

    events = list(event_to_files.keys())

    # Sort events by size (largest first)
    random.seed(seed)
    random.shuffle(events)  # randomize before sorting
    events.sort(key=lambda e: len(event_to_files[e]), reverse=True)

    total_tiles = len(imgs)
    target_train = 0.8 * total_tiles
    target_val = 0.1 * total_tiles
    target_test = 0.1 * total_tiles

    splits = {"train": [], "val": [], "test": []}
    split_sizes = {"train": 0, "val": 0, "test": 0}
    targets = {"train": target_train, "val": target_val, "test": target_test}

    for event in events:
        event_size = len(event_to_files[event])

        # assign to split with most remaining capacity
        best_split = min(
            splits.keys(),
            key=lambda s: split_sizes[s] / targets[s]
        )

        splits[best_split].append(event)
        split_sizes[best_split] += event_size

    train_events = splits["train"]
    val_events = splits["val"]
    test_events = splits["test"]

    # File Lists
    train_files = [f for e in train_events for f in event_to_files[e]]
    val_files = [f for e in val_events for f in event_to_files[e]]
    test_files = [f for e in test_events for f in event_to_files[e]]

    # Sanity Check (This is fine events do not have same # of tiles)
    print("Train:", len(train_files))
    print("Val:", len(val_files))
    print("Test:", len(test_files))

    return train_files, val_files, test_files