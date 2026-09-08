#!/usr/bin/env python3
"""
Copy paired image and mask files based on a two-column CSV mapping.
"""

import argparse
import csv
import sys
from pathlib import Path
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy images and masks according to a CSV mapping."
    )
    parser.add_argument(
        '--csv', '-c', required=True, type=Path,
        help='Path to input CSV file (two columns: image_relative_path,mask_relative_path).'
    )
    parser.add_argument(
        '--src_images', required=True, type=Path,
        help='Root directory for source images (prefix for first CSV column).'
    )
    parser.add_argument(
        '--src_masks', required=True, type=Path,
        help='Root directory for source masks (prefix for second CSV column).'
    )
    parser.add_argument(
        '--dst_images', required=True, type=Path,
        help='Destination directory for copied images.'
    )
    parser.add_argument(
        '--dst_masks', required=True, type=Path,
        help='Destination directory for copied masks.'
    )
    return parser.parse_args()

def copy_files(csv_path: Path, src_images: Path, src_masks: Path,
               dst_images: Path, dst_masks: Path):
    # Ensure destination directories exist
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_masks.mkdir(parents=True, exist_ok=True)

    with csv_path.open('r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue  # Skip malformed lines
            image_rel, mask_rel = row[0].strip(), row[1].strip()

            src_image_path = src_images / image_rel
            src_mask_path = src_masks / mask_rel

            if not src_image_path.is_file():
                print(f"[ERROR] Image not found: {src_image_path}", file=sys.stderr)
                continue
            if not src_mask_path.is_file():
                print(f"[ERROR] Mask not found:  {src_mask_path}", file=sys.stderr)
                continue

            # Keep only the filename, copy into the destination directory
            dst_image_path = dst_images / Path(image_rel).name
            dst_mask_path = dst_masks / Path(mask_rel).name

            shutil.copy2(src_image_path, dst_image_path)
            shutil.copy2(src_mask_path, dst_mask_path)
            print(f"[OK] Copied image: {src_image_path} → {dst_image_path}")
            print(f"[OK] Copied mask:  {src_mask_path} → {dst_mask_path}")

def main():
    args = parse_args()
    copy_files(
        csv_path   = args.csv,
        src_images = args.src_images,
        src_masks  = args.src_masks,
        dst_images = args.dst_images,
        dst_masks  = args.dst_masks,
    )

if __name__ == "__main__":
    main()
