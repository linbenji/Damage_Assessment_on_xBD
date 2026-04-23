# Siamese U-Net for Building Damage Assessment on xBD

**Semantic segmentation of building damage severity from pre- and post-disaster satellite imagery.**

John Creighton & Benjamin Lin — Khoury College of Computer Sciences, Northeastern University

---

## Overview

This project develops and evaluates CNN-based architectures for pixel-level building damage assessment using the [xBD dataset](https://xview2.org/). Given a pair of pre- and post-disaster satellite images, the model produces a per-pixel damage map classifying each pixel into one of five categories: background, no damage, minor damage, major damage, or destroyed.

Two core architectures are implemented and compared:
- **Baseline U-Net** — pre/post images concatenated as a 6-channel input
- **Siamese U-Net** — shared-weight encoder with multi-scale feature fusion, explicitly modeling temporal change between image pairs

---

## Repository Structure

```
├── src/
│   ├── model_unet.py                    # Baseline U-Net architecture
│   ├── model_siamese.py                 # Siamese U-Net architecture
│   ├── train.py                         # Training loop, losses, early stopping
│   ├── eval.py                          # Evaluation utilities, confusion matrix tracker
│   ├── augment.py                       # Dataset class with geometric augmentations and oversampling
│   ├── dataloader.py                    # DataLoader construction with WeightedRandomSampler
│   ├── xBD_splits.py                    # Event-based train/val/test splitting
│   └── __init__.py
├── Model_Architectures Notebooks/
│   ├── baseline_UNet.ipynb              # UNet with CE loss only
│   ├── baseline_UNet_v2.ipynb           # UNet with ComboLoss (Tversky + weighted CE)
│   ├── baseline_Siamese_UNet.ipynb      # Siamese UNet baseline
│   ├── custom_weight_Siamese_UNet.ipynb # Siamese UNet with inverse effective number class weights
│   ├── ohem_Siamese_UNet.ipynb          # Siamese UNet with OHEM
│   └── Results Comparison.ipynb         # Aggregated results, metrics comparison, plotting
├── Preprocessing
│   ├──tiling.py                            # GeoTIFF tiling and mask rasterization pipeline
│   ├──xBD_EDA.ipynb                        # EDA of xBD Dataset
│   ├──xBD_Preprocess_Images.ipynb          # Tile Generation and Sanity Checks 
│   ├──xBD_splits.py                        # Train/Val/Test Splitting Functions 
│   ├──tiles/   
│       ├── images/                          # Tiled image patches (generated)
│       └── masks/                           # Tiled mask patches (generated)
│── .gitignore
│── LICENSE
└── README.md
```

---

## Dataset

This project uses the [xBD dataset](https://xview2.org/), which must be downloaded separately.

```
xBD/
├── images/
│   ├── <disaster>_00000000_pre_disaster.tif
│   ├── <disaster>_00000000_post_disaster.tif
│   └── ...
└── labels/
    ├── <disaster>_00000000_post_disaster.json
    └── ...
```

**Class distribution (severe imbalance):**

| Class | Label | Pixel % |
|---|---|---|
| Background | 0 | 90.78% |
| No Damage | 1 | 8.02% |
| Minor Damage | 2 | 0.42% |
| Major Damage | 3 | 0.53% |
| Destroyed | 4 | 0.25% |

---

## Setup

### Requirements

```bash
pip install torch torchvision rasterio shapely Pillow numpy matplotlib tqdm
```

### 1. Tile the Dataset

Run `tiling.py` on each image/label pair to generate 512×512 patches:

```python
from tiling import tile_image

tile_image(
    image_path="xBD/images/hurricane-harvey_00000001_post_disaster.tif",
    label_path="xBD/labels/hurricane-harvey_00000001_post_disaster.json",
    skip_empty=True   # drops tiles with no labeled pixels
)
# Output saved to tiles/images/ and tiles/masks/
```

Each 1024×1024 scene is tiled into a 2×2 grid, yielding up to 4 patches per image. After discarding empty tiles, 6,846 usable pre/post tile pairs remain from the 2,799 original image pairs.

### 2. Generate Splits

Splits are performed at the **disaster event level** (not tile level) to prevent data leakage:

```python
from src.xBD_splits import create_splits

train_files, val_files, test_files = create_splits(
    img_dir="tiles/images/",
    mask_dir="tiles/masks/",
    seed=42  # 80% train, 10% val, 10% test
)
```

### 3. Build DataLoaders

```python
from src.dataloader import get_loaders

train_loader, val_loader, test_loader = get_loaders(
    train_files, val_files, test_files,
    img_dir="tiles/images/",
    mask_dir="tiles/masks/",
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    siamese=False,       # True for Siamese UNet
    augment_train=True   # horizontal/vertical flips, 90° rotations
)
```

Oversampling weights are automatically computed at dataset init and passed to `WeightedRandomSampler`:

| Tile contains... | Sampling weight |
|---|---|
| Destroyed (class 4) | 15.0 |
| Major (class 3) | 10.0 |
| Minor (class 2) | 3.0 |
| No rare classes | 1.0 |

---

## Training

### Baseline U-Net

```python
import torch
from src.model_unet import UNet
from src.train import run_training, ComboLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(in_channels=6, num_classes=5, base_features=32).to(device)

class_weights = torch.tensor([...]).to(device)  # see Class Weights section below
criterion = ComboLoss(class_weights=class_weights, ce_weight=0.5, tversky_weight=0.5,
                      tversky_alpha=0.7, tversky_beta=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

history = run_training(
    model, train_loader, val_loader,
    criterion, optimizer, scheduler, device,
    num_epochs=50, patience=12,
    save_path="best_unet.pth"
)
```

### Siamese U-Net

```python
from src.model_siamese import SiameseUNet

model = SiameseUNet(num_classes=5, base_features=32, in_channels=3).to(device)

# Use siamese=True in get_loaders — model receives (pre, post, mask) tuples
```

### Class Weights (Inverse Effective Number)

```python
beta = 0.9999
class_counts = torch.tensor([
    3258249517,   # background
     287893139,   # no damage
      15076019,   # minor
      19138232,   # major
       8918741    # destroyed
], dtype=torch.float32)

effective_num = (1.0 - beta ** class_counts) / (1.0 - beta)
weights = 1.0 / effective_num
weights = weights / weights.min()
class_weights = weights.to(device)
```

---

## Evaluation

```python
from src.eval import test_evaluation

test_loss, test_metrics = test_evaluation(
    model, test_loader, criterion, device,
    save_path="best_unet.pth",
    num_classes=5
)
```

Outputs per-class IoU, Precision, Recall, mIoU, and overall accuracy.

### Plot Training History

```python
from src.train import plot_train_history
plot_train_history(history)  # loss curves, mIoU curves, LR schedule
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Event-based splits | Prevents tile-level data leakage across train/test from the same disaster |
| Skip empty tiles | Removes background-only patches that provide no useful gradient signal |
| Tile-level oversampling | Increases frequency of rare-class tiles in each training epoch |
| Tversky loss (α=0.7) | Penalizes missed detections of rare damage classes more than false alarms |
| Inverse effective number weights | Robust class weighting for extreme pixel count disparities |
| Triple fusion in Siamese | Encodes pre context, post context, and change magnitude simultaneously |

---

## Limitations

- **Rare class sparsity** — Major and Destroyed pixels comprise <1% of all pixels. Even with oversampling, individual batches contain very few damaged pixels, limiting gradient signal.
- **No global context** — 512×512 tiling removes scene-level spatial context about disaster extent.
- **Training from scratch** — No pretrained encoder weights; the Siamese architecture in particular would benefit from a pretrained backbone.

---

## Citation

If you use this code, please cite the xBD dataset:

```bibtex
@article{gupta2019xbd,
  title={xBD: A Dataset for Assessing Building Damage from Satellite Imagery},
  author={Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and
          Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew},
  journal={arXiv preprint arXiv:1911.09296},
  year={2019}
}
```