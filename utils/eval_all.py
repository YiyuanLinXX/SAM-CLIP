#!/usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

# ANSI color codes
COLOR_GREEN  = "\033[92m"
COLOR_CYAN   = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET  = "\033[0m"

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS


# ------------------ label scanning and mapping ------------------ #

def scan_unique_labels(pred_files, gt_files_map):
    unique_values = set()

    for pred_file in tqdm(pred_files, desc="Scanning labels", leave=False):
        stem = pred_file.stem
        gt_path = gt_files_map.get(stem, None)
        if gt_path is None:
            continue

        pred = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or label is None:
            continue
        if pred.shape != label.shape:
            continue

        unique_values.update(np.unique(pred))
        unique_values.update(np.unique(label))

    if not unique_values:
        unique_values = {0}
        print(f"{COLOR_YELLOW}No valid labels found; fallback to single class [0].{COLOR_RESET}")

    label_values = sorted(int(v) for v in unique_values)
    print(f"{COLOR_CYAN}Detected label values: {label_values}{COLOR_RESET}")
    return label_values


def build_lut(label_values):
    value_to_index = {v: i for i, v in enumerate(label_values)}

    max_raw = max(label_values)
    lut = np.full(max_raw + 1, -1, dtype=np.int32)
    for v, idx in value_to_index.items():
        if v < 0:
            raise ValueError("Negative label values are not supported.")
        lut[v] = idx

    return lut, value_to_index


def encode_with_lut(mask, lut):
    mask_int = mask.astype(np.int32)
    max_allowed = lut.shape[0] - 1
    if mask_int.max() > max_allowed:
        raise ValueError(
            f"Found label value {mask_int.max()} larger than LUT max index {max_allowed}."
        )
    encoded = lut[mask_int]
    encoded[encoded < 0] = 0
    return encoded


# ------------------ IoU utilities ------------------ #

def intersect_and_union(pred_idx, label_idx, num_classes):
    pred_flat = pred_idx.ravel()
    label_flat = label_idx.ravel()
    intersect = pred_flat[pred_flat == label_flat]

    area_intersect = np.bincount(intersect, minlength=num_classes)
    area_pred      = np.bincount(pred_flat, minlength=num_classes)
    area_label     = np.bincount(label_flat, minlength=num_classes)
    area_union     = area_pred + area_label - area_intersect

    return area_intersect.astype(np.float64), area_union.astype(np.float64)


def compute_iou(area_intersect, area_union):
    return area_intersect / (area_union + 1e-10)


# ------------------ Main evaluation ------------------ #

def evaluate_folders(
    pred_folder,
    gt_folder,
    output_csv,
    save_per_image=False,
    quantiles=[1, 5, 10],
    target_labels=None,
):

    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)

    pred_files = sorted([p for p in pred_folder.iterdir() if is_image_file(p)])
    gt_files = [p for p in gt_folder.iterdir() if is_image_file(p)]
    gt_files_map = {f.stem: f for f in gt_files}

    if len(pred_files) == 0:
        print(f"{COLOR_YELLOW}No prediction images found in {pred_folder}.{COLOR_RESET}")
        return

    # 1) Collect raw label values from all images
    label_values = scan_unique_labels(pred_files, gt_files_map)
    lut, value_to_index = build_lut(label_values)
    num_classes = len(label_values)

    print(f"{COLOR_CYAN}Number of classes = {num_classes}{COLOR_RESET}")
    print(f"{COLOR_CYAN}Mapping (index -> raw): { {i: v for i, v in enumerate(label_values)} }{COLOR_RESET}")

    # Decide target_labels
    if target_labels is None or len(target_labels) == 0:
        # auto: all non-minimum labels
        if len(label_values) >= 2:
            target_labels = label_values[1:]
        else:
            target_labels = label_values[:]
        print(f"{COLOR_CYAN}Auto-selected target_labels = {target_labels}{COLOR_RESET}")

    filtered_targets = []
    for t in target_labels:
        if t in value_to_index:
            filtered_targets.append(t)
        else:
            print(f"{COLOR_YELLOW}Warning: target_label {t} not found in dataset; skipped.{COLOR_RESET}")

    target_labels = filtered_targets
    target_class_indices = [value_to_index[t] for t in target_labels]

    # Allocations
    total_area_intersect = np.zeros(num_classes, dtype=np.float64)
    total_area_union     = np.zeros(num_classes, dtype=np.float64)

    # For correct per-class image-avg IoU
    per_class_iou_sums = np.zeros(num_classes, dtype=np.float64)
    per_class_counts   = np.zeros(num_classes, dtype=np.float64)

    images_counted = 0

    # Store per-image data
    per_image_ious_overall = []
    per_image_ious_per_class = []
    per_image_records = []

    missing_count = 0

    # 2) Main loop
    for pred_file in tqdm(pred_files, desc="Evaluating"):
        stem = pred_file.stem
        gt_path = gt_files_map.get(stem, None)
        if gt_path is None:
            print(f"{COLOR_YELLOW}Warning: No GT found for {pred_file.name}{COLOR_RESET}")
            missing_count += 1
            continue

        pred = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or label is None:
            continue
        if pred.shape != label.shape:
            continue

        pred_idx = encode_with_lut(pred, lut)
        label_idx = encode_with_lut(label, lut)

        area_int, area_uni = intersect_and_union(pred_idx, label_idx, num_classes)

        # Dataset-level stats
        total_area_intersect += area_int
        total_area_union     += area_uni

        # Compute per-class IoU for this image
        iou_per_class = compute_iou(area_int, area_uni)

        # Only count this image for class c if class appears
        for c in range(num_classes):
            if area_uni[c] > 0:
                per_class_iou_sums[c] += iou_per_class[c]
                per_class_counts[c] += 1

        images_counted += 1

        # Store per-image
        img_overall_iou = float(np.mean(iou_per_class))
        per_image_ious_overall.append((pred_file.name, img_overall_iou))
        per_image_ious_per_class.append((pred_file.name, iou_per_class.copy()))
        if save_per_image:
            per_image_records.append({
                "Filename": pred_file.name,
                "IoU_per_image_overall(%)": round(img_overall_iou * 100, 2)
            })

    # 3) Compute mIoU_D and mIoU_I
    iou_global_per_class = compute_iou(total_area_intersect, total_area_union)
    mIoU_D_overall = float(np.mean(iou_global_per_class))

    # per class image-average IoU (correct behavior)
    iou_imageavg_per_class = per_class_iou_sums / np.maximum(per_class_counts, 1e-10)
    mIoU_I_overall = float(np.mean(iou_imageavg_per_class))

    # 4) Worst case metrics
    worst_overall = []
    worst_targets = {}

    if per_image_ious_overall:
        per_image_ious_overall.sort(key=lambda x: x[1])
        total = len(per_image_ious_overall)
        for q in quantiles:
            k = max(1, int(total * q / 100))
            worst_cases = per_image_ious_overall[:k]
            avg_iou = float(np.mean([x[1] for x in worst_cases]))
            worst_overall.append((q, k, avg_iou, worst_cases))

    # for each target label
    for t, cls_idx in zip(target_labels, target_class_indices):
        lst = [(fn, float(vec[cls_idx])) for fn, vec in per_image_ious_per_class]
        lst.sort(key=lambda x: x[1])
        total_t = len(lst)

        t_blocks = []
        for q in quantiles:
            k = max(1, int(total_t * q / 100))
            worst_cases = lst[:k]
            avg_iou = float(np.mean([x[1] for x in worst_cases]))
            t_blocks.append((q, k, avg_iou, worst_cases))

        worst_targets[t] = t_blocks

    # ------------------ CSV ------------------ #

    blocks = []

    blocks.append(pd.DataFrame({
        "ClassIndex": list(range(num_classes)),
        "RawLabelValue": label_values,
        "IoU_Global(%)": np.round(iou_global_per_class * 100, 2),
        "IoU_ImageAvg(%)": np.round(iou_imageavg_per_class * 100, 2),
    }))

    blocks.append(pd.DataFrame({
        "Metric": ["mIoU_D_overall(%)", "mIoU_I_overall(%)"],
        "Value":  [round(mIoU_D_overall * 100, 2), round(mIoU_I_overall * 100, 2)]
    }))

    if worst_overall:
        rows = []
        for q, k, avg_iou, worst_cases in worst_overall:
            rows.append({
                "WorstCase_Overall": f"Worst {q}%",
                "NumImages": k,
                "AvgIoU_overall(%)": round(avg_iou * 100, 2),
                "Filenames": "; ".join([wc[0] for wc in worst_cases])
            })
        blocks.append(pd.DataFrame(rows))

    for t in target_labels:
        t_blocks = worst_targets[t]
        rows_t = []
        for q, k, avg_iou, worst_cases in t_blocks:
            rows_t.append({
                "WorstCase_TargetLabel": f"Worst {q}%",
            "TargetLabel": t,
            "NumImages": k,
            "AvgIoU_target_label(%)": round(avg_iou * 100, 2),
            "Filenames": "; ".join([wc[0] for wc in worst_cases])
        })
        blocks.append(pd.DataFrame(rows_t))

    if save_per_image and per_image_records:
        blocks.append(pd.DataFrame(per_image_records))

    with open(output_csv, "w") as f:
        for idx, block in enumerate(blocks):
            block.to_csv(f, index=False)
            if idx < len(blocks) - 1:
                f.write("\n")

    # ------------------ print summary ------------------ #

    print(f"\nResults saved to: {output_csv}\n")

    print("Per-class mIoU:")
    for i in range(num_classes):
        raw = label_values[i]
        print(
            f"  ClassIndex={i}, RawLabel={raw}: "
            f"{COLOR_GREEN}mIoU_D={iou_global_per_class[i]*100:.2f}%{COLOR_RESET} "
            f"{COLOR_CYAN}mIoU_I={iou_imageavg_per_class[i]*100:.2f}%{COLOR_RESET}"
        )

    print(
        f"\nOverall: {COLOR_GREEN}mIoU_D={mIoU_D_overall*100:.2f}%{COLOR_RESET}, "
        f"{COLOR_CYAN}mIoU_I={mIoU_I_overall*100:.2f}%{COLOR_RESET}"
    )

    if worst_overall:
        print(f"\n{COLOR_YELLOW}=== Worst Case (Overall IoU) ==={COLOR_RESET}")
        for q, k, avg_iou, _ in worst_overall:
            print(f"Worst {q}% ({k} images): Avg IoU = {avg_iou*100:.2f}%")

    for t in target_labels:
        print(f"\n{COLOR_YELLOW}=== Worst Case for Target Label {t} ==={COLOR_RESET}")
        for q, k, avg_iou, _ in worst_targets[t]:
            print(f"Worst {q}% ({k} images): Avg IoU = {avg_iou*100:.2f}%")

    if missing_count > 0:
        print(f"{COLOR_YELLOW}{missing_count} prediction image(s) had no matching GT image.{COLOR_RESET}")


# ------------------ CLI ------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-class mIoU_D (global) and mIoU_I (image avg), "
            "plus worst-case IoU (overall and per target label). "
            "Distinct raw pixel values are treated as classes."
        )
    )
    parser.add_argument("--pred_folder", type=str, required=True)
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--save_per_image", action="store_true")
    parser.add_argument("--quantiles", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--target_labels",
        type=int,
        nargs="+",
        default=None,
        help="Raw pixel values for worst-case analysis. Leave empty for auto."
    )

    args = parser.parse_args()

    evaluate_folders(
        pred_folder=args.pred_folder,
        gt_folder=args.gt_folder,
        output_csv=args.output_csv,
        save_per_image=args.save_per_image,
        quantiles=args.quantiles,
        target_labels=args.target_labels,
    )

