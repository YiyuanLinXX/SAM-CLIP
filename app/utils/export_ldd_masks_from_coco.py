#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BASE_PRIORITY = {
    "vines_leaf": 0,
    "vines_grape": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export semantic mask PNGs from the LDD COCO annotations."
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to a COCO-style annotation json file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save exported mask PNGs.",
    )
    parser.add_argument(
        "--mapping-path",
        default=None,
        help="Optional path to save category-to-pixel metadata as JSON.",
    )
    return parser.parse_args()


def polygon_to_points(segmentation):
    return [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]


def draw_annotation(mask, annotation, pixel_value):
    segmentation = annotation.get("segmentation")
    if not isinstance(segmentation, list):
        return False

    overlay = Image.new("L", (mask.shape[1], mask.shape[0]), 0)
    drawer = ImageDraw.Draw(overlay)
    drew_anything = False

    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2 != 0:
            continue
        drawer.polygon(polygon_to_points(polygon), fill=int(pixel_value))
        drew_anything = True

    if not drew_anything:
        return False

    overlay_array = np.array(overlay, dtype=np.uint8)
    positive = overlay_array > 0
    mask[positive] = overlay_array[positive]
    return True


def build_priority(category_name):
    return BASE_PRIORITY.get(category_name, 1)


def main():
    args = parse_args()

    annotation_path = Path(args.annotation)
    output_dir = Path(args.output_dir)
    mapping_path = Path(args.mapping_path) if args.mapping_path else output_dir / "label_mapping.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    with annotation_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda item: item["id"])
    category_name_by_id = {item["id"]: item["name"] for item in categories}
    pixel_value_by_category_id = {item["id"]: item["id"] + 1 for item in categories}
    priority_by_category_id = {
        item["id"]: build_priority(item["name"]) for item in categories
    }

    annotations_by_image_id = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation.get("segmentation"):
            annotations_by_image_id[annotation["image_id"]].append(annotation)

    exported = 0
    for image_info in data["images"]:
        image_id = image_info["id"]
        height = image_info["height"]
        width = image_info["width"]
        mask = np.zeros((height, width), dtype=np.uint8)

        annotations = annotations_by_image_id.get(image_id, [])
        annotations.sort(
            key=lambda item: (
                priority_by_category_id[item["category_id"]],
                -float(item.get("area", 0.0)),
                item["id"],
            )
        )

        for annotation in annotations:
            pixel_value = pixel_value_by_category_id[annotation["category_id"]]
            draw_annotation(mask, annotation, pixel_value)

        out_name = Path(image_info["file_name"]).stem + ".png"
        Image.fromarray(mask, mode="L").save(output_dir / out_name)
        exported += 1

    mapping = {
        "background": 0,
        "annotation_file": str(annotation_path),
        "num_images": len(data["images"]),
        "num_annotations": len(data["annotations"]),
        "labels": [
            {
                "category_id": item["id"],
                "category_name": item["name"],
                "pixel_value": pixel_value_by_category_id[item["id"]],
                "priority": priority_by_category_id[item["id"]],
            }
            for item in categories
        ],
    }

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Exported {exported} masks to {output_dir}")
    print(f"Saved label mapping to {mapping_path}")


if __name__ == "__main__":
    main()
