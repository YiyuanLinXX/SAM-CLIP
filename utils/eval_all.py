#!/usr/bin/env python3
# coding=utf-8

import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from pathlib import Path

# === ANSI color codes ===
COLOR_GREEN  = "\033[92m"
COLOR_CYAN   = "\033[96m"
COLOR_YELLOW = "\033[93m"
COLOR_RESET  = "\033[0m"

# Only consider these as images (lowercase suffixes)
ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_EXTS

def intersect_and_union(pred, label, num_classes=2, ignore_index=255):
    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]

    intersect = pred[pred == label]
    area_intersect = np.histogram(intersect, bins=num_classes, range=(0, num_classes))[0]
    area_pred      = np.histogram(pred,      bins=num_classes, range=(0, num_classes))[0]
    area_label     = np.histogram(label,     bins=num_classes, range=(0, num_classes))[0]
    area_union     = area_pred + area_label - area_intersect
    return area_intersect, area_union

def compute_iou(area_intersect, area_union):
    return area_intersect / (area_union + 1e-10)

def evaluate_folders(pred_folder, gt_folder, output_csv,
                     save_per_image=False, num_classes=2, ignore_index=255,
                     quantiles=[1,5,10], target_class=1):

    pred_folder = Path(pred_folder)
    gt_folder   = Path(gt_folder)

    # Only load image files
    pred_files = sorted([p for p in pred_folder.iterdir() if is_image_file(p)])
    gt_files   = [p for p in gt_folder.iterdir() if is_image_file(p)]
    gt_files_map = {f.stem: f for f in gt_files}  # stem -> Path

    # —— For mIoU_D (global, per-class) ——
    total_area_intersect = np.zeros(num_classes, dtype=np.float64)
    total_area_union     = np.zeros(num_classes, dtype=np.float64)

    # —— For mIoU_I (image-avg, per-class) ——
    per_class_iou_sums   = np.zeros(num_classes, dtype=np.float64)
    images_counted       = 0

    # Per-image IoUs
    per_image_ious_overall = []  # (filename, overall_iou)
    per_image_ious_per_class = []  # (filename, iou_vector[num_classes])
    per_image_records = []

    missing_count = 0

    for pred_file in tqdm(pred_files, desc="Evaluating"):
        stem = pred_file.stem
        gt_path = gt_files_map.get(stem, None)
        if gt_path is None:
            print(f"{COLOR_YELLOW}Warning: No GT found for {pred_file.name}{COLOR_RESET}")
            missing_count += 1
            continue

        pred  = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(gt_path),   cv2.IMREAD_GRAYSCALE)
        if pred is None or label is None:
            print(f"{COLOR_YELLOW}Warning: Cannot read {pred_file.name} or its GT. Skipping.{COLOR_RESET}")
            continue

        # Binary 0/255 -> 0/1  (二分类假设)
        pred  = (pred  == 255).astype(np.uint8)
        label = (label == 255).astype(np.uint8)

        # Dataset-level accumulation
        area_int, area_uni = intersect_and_union(pred, label, num_classes, ignore_index)
        total_area_intersect += area_int
        total_area_union     += area_uni

        # Image-level IoU (per-class)
        iou_per_class = compute_iou(area_int, area_uni)  # shape: (num_classes,)
        per_class_iou_sums += iou_per_class
        images_counted += 1

        # store per-image
        img_overall_iou = float(np.mean(iou_per_class))
        per_image_ious_overall.append((pred_file.name, img_overall_iou))
        per_image_ious_per_class.append((pred_file.name, iou_per_class.copy()))
        if save_per_image:
            per_image_records.append({
                "Filename": pred_file.name,
                "IoU_per_image_overall(%)": round(img_overall_iou * 100, 2)
            })

    # —— mIoU_D (global, per-class & overall) ——
    iou_global_per_class = compute_iou(total_area_intersect, total_area_union)
    mIoU_D_overall = float(np.mean(iou_global_per_class))

    # —— mIoU_I (image-avg, per-class & overall) ——
    if images_counted > 0:
        iou_imageavg_per_class = per_class_iou_sums / images_counted
        mIoU_I_overall = float(np.mean(iou_imageavg_per_class))
    else:
        iou_imageavg_per_class = np.zeros(num_classes, dtype=np.float64)
        mIoU_I_overall = 0.0

    # —— Worst Case (two tracks) ——
    worst_overall = []       # based on overall per-image IoU
    worst_target  = []       # based on target-class per-image IoU

    if per_image_ious_overall:
        # Overall
        per_image_ious_overall.sort(key=lambda x: x[1])  # ascending
        total = len(per_image_ious_overall)
        for q in quantiles:
            k = max(1, int(total * q / 100))
            worst_cases = per_image_ious_overall[:k]
            avg_iou = float(np.mean([x[1] for x in worst_cases]))
            worst_overall.append((q, k, avg_iou, worst_cases))

    if per_image_ious_per_class:
        # Target-class
        # build (filename, iou_target)
        ti = int(target_class)
        cls_list = [(fn, float(vec[ti])) for fn, vec in per_image_ious_per_class]
        cls_list.sort(key=lambda x: x[1])  # ascending
        total_t = len(cls_list)
        for q in quantiles:
            k = max(1, int(total_t * q / 100))
            worst_cases = cls_list[:k]
            avg_iou = float(np.mean([x[1] for x in worst_cases]))
            worst_target.append((q, k, avg_iou, worst_cases))

    # —— CSV blocks ——
    blocks = []

    # Block 1: per-class (two methods)
    block1 = pd.DataFrame({
        "Class": [f"Class_{i}" for i in range(num_classes)],
        "IoU_Global(%)":   np.round(iou_global_per_class   * 100, 2),  # mIoU_D_class
        "IoU_ImageAvg(%)": np.round(iou_imageavg_per_class * 100, 2),  # mIoU_I_class
    })
    blocks.append(block1)

    # Block 2: overall summary
    block2 = pd.DataFrame({
        "Metric": ["mIoU_D_overall(%)", "mIoU_I_overall(%)"],
        "Value":  [round(mIoU_D_overall*100, 2), round(mIoU_I_overall*100, 2)]
    })
    blocks.append(block2)

    # Block 3: worst case - overall
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

    # Block 4: worst case - target class
    if worst_target:
        rows_t = []
        for q, k, avg_iou, worst_cases in worst_target:
            rows_t.append({
                "WorstCase_TargetClass": f"Worst {q}%",
                "TargetClass": target_class,
                "NumImages": k,
                "AvgIoU_target_class(%)": round(avg_iou * 100, 2),
                "Filenames": "; ".join([wc[0] for wc in worst_cases])
            })
        blocks.append(pd.DataFrame(rows_t))

    # Block 5: per-image overall IoU (optional)
    if save_per_image and per_image_records:
        blocks.append(pd.DataFrame(per_image_records))

    # Write CSV
    with open(output_csv, "w") as f:
        for idx, block in enumerate(blocks):
            block.to_csv(f, index=False)
            if idx < len(blocks) - 1:
                f.write("\n")

    # —— terminal output ——
    print(f"\n✅ Results saved to: {output_csv}")

    print(f"\nPer-class mIoU (two methods):")
    for i in range(num_classes):
        print(
            f"  Class_{i}: "
            f"{COLOR_GREEN}mIoU_D_class={iou_global_per_class[i]*100:.2f}%{COLOR_RESET}  "
            f"{COLOR_CYAN}mIoU_I_class={iou_imageavg_per_class[i]*100:.2f}%{COLOR_RESET}"
        )

    print(
        f"\nOverall: "
        f"{COLOR_GREEN}mIoU_D={mIoU_D_overall*100:.2f}%{COLOR_RESET}, "
        f"{COLOR_CYAN}mIoU_I={mIoU_I_overall*100:.2f}%{COLOR_RESET}"
    )

    if worst_overall:
        print(f"\n{COLOR_YELLOW}=== Worst Case (Overall IoU) ==={COLOR_RESET}")
        for q, k, avg_iou, _ in worst_overall:
            print(f"Worst {q}% ({k} images) Avg Overall IoU: {avg_iou*100:.2f}%")

    if worst_target:
        print(f"\n{COLOR_YELLOW}=== Worst Case (Target Class IoU, class={target_class}) ==={COLOR_RESET}")
        for q, k, avg_iou, _ in worst_target:
            print(f"Worst {q}% ({k} images) Avg Target-Class IoU: {avg_iou*100:.2f}%")

    if missing_count > 0:
        print(f"{COLOR_YELLOW}⚠️ {missing_count} prediction image(s) had no matching GT image.{COLOR_RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute per-class mIoU_D (global) & mIoU_I (image-avg), overall metrics, and worst-case IoU (overall & target-class) for binary segmentation."
    )
    parser.add_argument("--pred_folder", type=str, required=True, help="Folder with predicted masks")
    parser.add_argument("--gt_folder",   type=str, required=True, help="Folder with ground truth masks")
    parser.add_argument("--output_csv",  type=str, required=True, help="Path to output CSV")
    parser.add_argument("--save_per_image", action="store_true", help="Save per-image overall IoU in the CSV")
    parser.add_argument("--quantiles", type=int, nargs="+", default=[1,5,10],
                        help="Percentages for worst case analysis (e.g., 1 5 10)")
    parser.add_argument("--target_class", type=int, default=1,
                        help="Target class index for worst-case-by-class analysis (default 1)")
    args = parser.parse_args()

    evaluate_folders(pred_folder=args.pred_folder,
                     gt_folder=args.gt_folder,
                     output_csv=args.output_csv,
                     save_per_image=args.save_per_image,
                     num_classes=2,  # binary segmentation
                     quantiles=args.quantiles,
                     target_class=args.target_class)

