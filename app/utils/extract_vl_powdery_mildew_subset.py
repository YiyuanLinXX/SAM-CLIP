#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


TARGET_CATEGORY_NAME = "vl_powdery_mildew"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract images containing vl_powdery_mildew and export binary masks."
    )
    parser.add_argument("--annotation", required=True, help="Path to COCO annotation json.")
    parser.add_argument("--images-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--semantic-mask-dir", required=True, help="Directory containing semantic mask PNGs.")
    parser.add_argument("--output-dir", required=True, help="Output directory for the subset.")
    return parser.parse_args()


def main():
    args = parse_args()

    annotation_path = Path(args.annotation)
    images_dir = Path(args.images_dir)
    semantic_mask_dir = Path(args.semantic_mask_dir)
    output_dir = Path(args.output_dir)
    out_images_dir = output_dir / "images"
    out_masks_dir = output_dir / "masks"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {item["id"]: item["name"] for item in data["categories"]}
    target_category_id = None
    for category_id, category_name in categories.items():
        if category_name == TARGET_CATEGORY_NAME:
            target_category_id = category_id
            break

    if target_category_id is None:
        raise ValueError(f"Category {TARGET_CATEGORY_NAME} not found in {annotation_path}")

    target_pixel_value = target_category_id + 1
    selected_image_ids = {
        annotation["image_id"]
        for annotation in data["annotations"]
        if annotation["category_id"] == target_category_id
    }

    exported = 0
    for image_info in data["images"]:
        if image_info["id"] not in selected_image_ids:
            continue

        image_name = image_info["file_name"]
        image_stem = Path(image_name).stem
        source_image_path = images_dir / image_name
        source_mask_path = semantic_mask_dir / f"{image_stem}.png"
        output_image_path = out_images_dir / image_name
        output_mask_path = out_masks_dir / f"{image_stem}.png"

        if not source_image_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_image_path}")
        if not source_mask_path.exists():
            raise FileNotFoundError(f"Missing semantic mask: {source_mask_path}")

        shutil.copy2(source_image_path, output_image_path)

        semantic_mask = np.array(Image.open(source_mask_path), dtype=np.uint8)
        binary_mask = np.where(semantic_mask == target_pixel_value, 255, 0).astype(np.uint8)
        Image.fromarray(binary_mask, mode="L").save(output_mask_path)
        exported += 1

    metadata = {
        "annotation_file": str(annotation_path),
        "source_images_dir": str(images_dir),
        "source_semantic_mask_dir": str(semantic_mask_dir),
        "target_category_name": TARGET_CATEGORY_NAME,
        "target_category_id": target_category_id,
        "target_pixel_value_in_semantic_mask": target_pixel_value,
        "binary_mask_foreground_value": 255,
        "num_images": exported,
    }

    with (output_dir / "subset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Exported {exported} images to {out_images_dir}")
    print(f"Exported {exported} binary masks to {out_masks_dir}")


if __name__ == "__main__":
    main()
