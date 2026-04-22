"""
visualization.py

Grid-based visualization utilities for xBD segmentation models.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


# Hard-code color map
COLORS = np.array([
    [0, 0, 0],        # No Damage
    [255, 255, 0],    # Minor
    [255, 165, 0],    # Major
    [255, 0, 0],      # Destroyed
    [128, 128, 128],  # Unclassified
], dtype=np.uint8)


def mask_to_rgb(mask):
    # Converts the mask to colors
    return COLORS[mask]

def denormalize(img):
    # Denormalizes images
    return (img * 0.5 + 0.5).clip(0, 1)


@torch.no_grad()
def visualize_predictions(model, test_loader, device, num_samples=4, title="Model Predictions", siamese = True):
    """
    Creates a grid of images, where each row is an example and the columns are
    Pre | Post | Ground Truth | Prediction
    """

    model.eval()

    fig, axes = plt.subplots(
        nrows=num_samples,
        ncols=4,
        figsize=(16, 4 * num_samples)
    )

    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    shown = 0

    for batch in test_loader:
        if siamese: 
            pre, post, masks = batch

            pre = pre.to(device)
            post = post.to(device)
            masks = masks.to(device)

            preds = model(pre, post).argmax(dim=1).cpu().numpy()

            pre = pre.cpu().numpy()
            post = post.cpu().numpy()
            masks = masks.cpu().numpy()
        else:
            images, masks = batch

            images = images.to(device)
            masks = masks.to(device)

            preds = model(images).argmax(dim=1).cpu().numpy()

            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            
        B = masks.shape[0]

        for i in range(B):
            if shown >= num_samples:
                plt.suptitle(title, fontsize=16)
                plt.tight_layout()
                plt.show()
                return

            # Images
            pre_img = np.transpose(denormalize(pre[i]), (1, 2, 0))
            post_img = np.transpose(denormalize(post[i]), (1, 2, 0))

            # Masks
            gt_mask = mask_to_rgb(masks[i])
            pred_mask = mask_to_rgb(preds[i])

            ax = axes[shown]

            ax[0].imshow(pre_img)
            ax[0].set_title("Pre")
            ax[0].axis("off")

            ax[1].imshow(post_img)
            ax[1].set_title("Post")
            ax[1].axis("off")

            ax[2].imshow(gt_mask)
            ax[2].set_title("Ground Truth")
            ax[2].axis("off")

            ax[3].imshow(pred_mask)
            ax[3].set_title("Prediction")
            ax[3].axis("off")

            shown += 1

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()