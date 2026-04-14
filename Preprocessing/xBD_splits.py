import os
from collections import defaultdict
import random

def extract_event(filename):
    return filename.split("_")[0]

def create_splits(img_dir, mask_dir, seed=42):

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
    random.seed(seed)
    random.shuffle(events)

    train_split = int(0.8 * len(events))
    val_split = int(0.9 * len(events))

    train_events = events[:train_split]
    val_events = events[train_split:val_split]
    test_events = events[val_split:]

    # File Lists
    train_files = [f for e in train_events for f in event_to_files[e]]
    val_files = [f for e in val_events for f in event_to_files[e]]
    test_files = [f for e in test_events for f in event_to_files[e]]

    # Sanity Check (This is fine events do not have same # of tiles)
    print("Train:", len(train_files))
    print("Val:", len(val_files))
    print("Test:", len(test_files))

    return train_files, val_files, test_files