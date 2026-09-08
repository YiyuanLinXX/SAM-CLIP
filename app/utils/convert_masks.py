#!/usr/bin/env python3
import os
import cv2
import argparse

def process_mask_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            continue

        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Read as grayscale
        mask = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: failed to read {in_path}")
            continue

        # Convert: 1 -> 255, 0 stays 0
        mask_out = (mask == 1).astype("uint8") * 255

        cv2.imwrite(out_path, mask_out)
        print(f"Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert mask values: 0 stay 0, 1 -> 255")
    parser.add_argument("--input-dir", required=True, help="Folder containing original masks")
    parser.add_argument("--output-dir", required=True, help="Folder to save converted masks")
    args = parser.parse_args()

    process_mask_folder(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
