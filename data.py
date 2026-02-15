import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import config
from model.MARCo_v4 import get_mask   # adjust path if needed


# =====================================
# SETTINGS
# =====================================
SAVE_DIR = "fig"
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_PATH = config.DATA_FILE
SPLIT = "train"        # "train" or "val"
INDEX = 0              # sample index

PATCH_SIZE = config.PATCH_SIZE
NUM_PATCHES = config.NUM_PATCHES
MASK_RATIO = config.MASK_RATIO
USE_8BIT = config.USE_8BIT


# =====================================
# SAME NORMALIZATION AS DATASET
# =====================================
def normalize_like_dataset(x, use_8bit=True):
    x = x.float()
    imgs = []

    for c in range(x.shape[0]):
        mean = x[c].mean()
        std = x[c].std()
        min_val = mean - 2 * std
        max_val = mean + 2 * std

        if use_8bit:
            img = (x[c] - min_val) / (max_val - min_val + 1e-6) * 255.0
            img = torch.clip(img, 0, 255).to(torch.uint8)
        else:
            img = (x[c] - min_val) / (max_val - min_val + 1e-6)
            img = torch.clip(img, 0, 1)

        imgs.append(img)

    return torch.stack(imgs, dim=0)


def to_float01(img):
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    img = img.float()
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    return img


def save_img(arr, path, cmap=None):
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def mask_to_image(mask_1d, H, W, patch_size):
    grid = int(np.sqrt(mask_1d.shape[0]))
    m = mask_1d.reshape(grid, grid)
    m_img = np.kron(m, np.ones((patch_size, patch_size)))
    return m_img[:H, :W]


# =====================================
# MAIN
# =====================================
def main():

    dfc = torch.load(DATA_PATH, map_location="cpu")

    if SPLIT.lower() == "train":
        images = dfc["train_images"]
    else:
        images = dfc["validation_images"]

    x = images[INDEX]  # (14, 96, 96)
    x = normalize_like_dataset(x, use_8bit=USE_8BIT)

    C, H, W = x.shape

    # Split modalities
    optical = x[:12]
    radar = x[12:]

    # Optical RGB (first 3 channels)
    optical_rgb = torch.stack([optical[0], optical[1], optical[2]], dim=0)
    optical_rgb = to_float01(optical_rgb).permute(1, 2, 0).numpy()

    # Radar grayscale (mean of 2 channels)
    radar_gray = to_float01(radar.float().mean(dim=0)).numpy()

    save_img(optical_rgb, os.path.join(SAVE_DIR, "optical_input.png"))
    save_img(radar_gray, os.path.join(SAVE_DIR, "radar_input.png"), cmap="gray")

    # =====================================
    # SHARED MASK (exactly like training)
    # =====================================
    shared_mask = get_mask(
        bsz=1,
        seq_len=NUM_PATCHES,
        device=torch.device("cpu"),
        mask_ratio=MASK_RATIO
    )

    mask_1d = shared_mask["mask_for_mae"][0].numpy()

    optical_mask_img = mask_to_image(mask_1d, H, W, PATCH_SIZE)
    radar_mask_img = mask_to_image(mask_1d, H, W, PATCH_SIZE)

    save_img(optical_mask_img, os.path.join(SAVE_DIR, "optical_mask.png"), cmap="gray")
    save_img(radar_mask_img, os.path.join(SAVE_DIR, "radar_mask.png"), cmap="gray")

    print("\nSaved figures to fig/:")
    print(" - optical_input.png")
    print(" - radar_input.png")
    print(" - optical_mask.png")
    print(" - radar_mask.png")


if __name__ == "__main__":
    main()
