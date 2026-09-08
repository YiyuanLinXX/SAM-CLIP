#!/usr/bin/env python3
"""
colorize_labelmasks.py

Convert single-channel label masks (0,1,2,3,4,...) into color masks
using a fixed color map (black, red, orange, yellow, green).
"""

import argparse
import os
import glob

import numpy as np
from PIL import Image


# Class id -> (R, G, B)
COLOR_MAP = {
    0: (0, 0, 0),          # background: black
    1: (0, 127, 0),        # class 1: green
    2: (216,191,216),      # class 2: purple
    3: (127, 0, 0),      # class 3: red
    4: (0, 0, 127),        # class 4: green
}
# Color for any id not in COLOR_MAP (optional)
UNKNOWN_COLOR = (255, 255, 255)  # white


def colorize_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    Convert a 2D label mask (H x W) to a color image (H x W x 3).
    """
    h, w = mask_arr.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    # First fill with UNKNOWN_COLOR
    if UNKNOWN_COLOR is not None:
        color_img[:, :] = np.array(UNKNOWN_COLOR, dtype=np.uint8)

    # Then overwrite with defined colors
    for cls_id, rgb in COLOR_MAP.items():
        mask = (mask_arr == cls_id)
        if not np.any(mask):
            continue
        color_img[mask] = np.array(rgb, dtype=np.uint8)

    return color_img


def main():
    parser = argparse.ArgumentParser(description="Colorize single-channel label masks.")
    parser.add_argument("--input-dir", required=True, help="Directory containing label-mask PNGs")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: INPUT/color_vis)")
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or os.path.join(input_dir, "color_vis")
    os.makedirs(output_dir, exist_ok=True)

    # Find all labelmask images
    pattern = os.path.join(input_dir, "*.png")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found with pattern: {pattern}")
        return

    print(f"Found {len(files)} label masks.")

    for path in files:
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)

        # Load single-channel mask
        mask = np.array(Image.open(path))

        if mask.ndim != 2:
            print(f"[WARN] {base} is not single-channel, skipping.")
            continue

        # Colorize
        color_img = colorize_mask(mask)

        # Save as *_vis.png
        save_name = f"{name}.png"
        save_path = os.path.join(output_dir, save_name)
        Image.fromarray(color_img).save(save_path)

        print(f"Saved color mask: {save_path}")

    print("Done. All color masks saved to:", output_dir)


if __name__ == "__main__":
    main()
