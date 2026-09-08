#!/usr/bin/env python3
# coding=utf-8

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTS


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return mask


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.float64)
    if union <= 0:
        return 0.0
    return float(inter / union)


def compute_ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1]))


def connected_components_for_class(mask: np.ndarray, raw_value: int):
    binary = (mask == raw_value).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for comp_id in range(1, num_labels):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        components.append(labels == comp_id)
    return components


def scan_label_values(semantic_dir: Path, gt_dir: Path, image_names):
    pred_values = set()
    gt_values = set()
    for image_name in tqdm(image_names, desc="Scanning labels", leave=False):
        pred = read_mask(semantic_dir / image_name)
        gt = read_mask(gt_dir / image_name)
        pred_values.update(int(v) for v in np.unique(pred))
        gt_values.update(int(v) for v in np.unique(gt))
    return sorted(pred_values), sorted(gt_values)


def infer_class_mappings(pred_dir: Path, gt_dir: Path, image_names, score_rows):
    semantic_dir = pred_dir / "semantic_masks"
    pred_raw_values, gt_raw_values = scan_label_values(semantic_dir, gt_dir, image_names)

    pred_fg_values = [v for v in pred_raw_values if v != 0]
    gt_fg_values = [v for v in gt_raw_values if v != 0]
    pred_class_ids = sorted({int(row["class_id"]) for row in score_rows if int(row["class_id"]) != 0})

    if len(pred_class_ids) != len(pred_fg_values):
        raise ValueError(
            "Mismatch between instance class ids and semantic-mask foreground classes: "
            f"class_ids={pred_class_ids}, semantic_fg_values={pred_fg_values}"
        )
    if len(pred_fg_values) != len(gt_fg_values):
        raise ValueError(
            "Mismatch between predicted semantic foreground classes and GT foreground classes: "
            f"pred_fg_values={pred_fg_values}, gt_fg_values={gt_fg_values}"
        )

    class_id_to_pred_raw = {cls_id: pred_raw for cls_id, pred_raw in zip(pred_class_ids, pred_fg_values)}
    pred_raw_to_gt_raw = {pred_raw: gt_raw for pred_raw, gt_raw in zip(pred_fg_values, gt_fg_values)}
    class_id_to_gt_raw = {
        cls_id: pred_raw_to_gt_raw[pred_raw]
        for cls_id, pred_raw in class_id_to_pred_raw.items()
    }
    return {
        "pred_raw_values": pred_raw_values,
        "gt_raw_values": gt_raw_values,
        "pred_fg_values": pred_fg_values,
        "gt_fg_values": gt_fg_values,
        "pred_class_ids": pred_class_ids,
        "class_id_to_pred_raw": class_id_to_pred_raw,
        "pred_raw_to_gt_raw": pred_raw_to_gt_raw,
        "class_id_to_gt_raw": class_id_to_gt_raw,
    }


def align_pred_to_gt_label_space(pred_mask: np.ndarray, pred_raw_to_gt_raw: dict) -> np.ndarray:
    aligned = np.zeros_like(pred_mask, dtype=np.uint8)
    for pred_raw, gt_raw in pred_raw_to_gt_raw.items():
        aligned[pred_mask == pred_raw] = gt_raw
    return aligned


def evaluate_semantic_miou(pred_dir: Path, gt_dir: Path, image_names, pred_raw_to_gt_raw: dict, eval_raw_values):
    semantic_dir = pred_dir / "semantic_masks"
    total_intersections = {raw: 0.0 for raw in eval_raw_values}
    total_unions = {raw: 0.0 for raw in eval_raw_values}
    per_image_rows = []

    for image_name in tqdm(image_names, desc="Computing semantic IoU", leave=False):
        pred_raw = read_mask(semantic_dir / image_name)
        gt_raw = read_mask(gt_dir / image_name)
        if pred_raw.shape != gt_raw.shape:
            raise ValueError(f"Shape mismatch for {image_name}: pred={pred_raw.shape}, gt={gt_raw.shape}")

        pred_aligned = align_pred_to_gt_label_space(pred_raw, pred_raw_to_gt_raw)
        valid_ious = []
        row = {"image_name": image_name}

        for raw_value in eval_raw_values:
            pred_bin = pred_aligned == raw_value
            gt_bin = gt_raw == raw_value
            inter = np.logical_and(pred_bin, gt_bin).sum(dtype=np.float64)
            union = np.logical_or(pred_bin, gt_bin).sum(dtype=np.float64)
            total_intersections[raw_value] += inter
            total_unions[raw_value] += union
            if union > 0:
                iou = float(inter / union)
                valid_ious.append(iou)
                row[f"class_{raw_value}_iou"] = iou
            else:
                row[f"class_{raw_value}_iou"] = np.nan

        row["image_miou"] = float(np.mean(valid_ious)) if valid_ious else np.nan
        per_image_rows.append(row)

    per_class_miou_d = {}
    for raw_value in eval_raw_values:
        union = total_unions[raw_value]
        per_class_miou_d[raw_value] = float(total_intersections[raw_value] / union) if union > 0 else np.nan

    image_mious = [row["image_miou"] for row in per_image_rows if not np.isnan(row["image_miou"])]
    miou_i = float(np.mean(image_mious)) if image_mious else np.nan

    valid_dataset_ious = [v for v in per_class_miou_d.values() if not np.isnan(v)]
    miou_d = float(np.mean(valid_dataset_ious)) if valid_dataset_ious else np.nan

    return {
        "miou_i": miou_i,
        "miou_d": miou_d,
        "per_class_miou_d": per_class_miou_d,
        "per_image_rows": per_image_rows,
    }


def build_gt_instances(gt_dir: Path, image_names, eval_raw_values):
    gt_instances = {raw_value: {} for raw_value in eval_raw_values}
    total_gt_by_class = {raw_value: 0 for raw_value in eval_raw_values}

    for image_name in tqdm(image_names, desc="Extracting GT instances", leave=False):
        gt_mask = read_mask(gt_dir / image_name)
        for raw_value in eval_raw_values:
            comps = connected_components_for_class(gt_mask, raw_value)
            gt_instances[raw_value][image_name] = comps
            total_gt_by_class[raw_value] += len(comps)

    return gt_instances, total_gt_by_class


def load_prediction_instances(pred_dir: Path, score_rows, class_id_to_gt_raw: dict):
    instance_dir = pred_dir / "instances"
    predictions_by_class = defaultdict(list)
    for row in tqdm(score_rows, desc="Loading predicted instances", leave=False):
        class_id = int(row["class_id"])
        if class_id == 0:
            continue
        gt_raw_value = class_id_to_gt_raw[class_id]
        instance_path = instance_dir / row["instance_mask"]
        pred_mask = read_mask(instance_path) > 0
        predictions_by_class[gt_raw_value].append({
            "image_name": row["image_name"],
            "instance_mask": row["instance_mask"],
            "score": float(row["score"]),
            "mask": pred_mask,
        })
    return predictions_by_class


def prepare_class_predictions(predictions, gt_instances_for_class):
    sorted_predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)
    prepared = []
    for pred in sorted_predictions:
        gt_list = gt_instances_for_class.get(pred["image_name"], [])
        candidates = []
        for gt_idx, gt_mask in enumerate(gt_list):
            iou = mask_iou(pred["mask"], gt_mask)
            if iou > 0.0:
                candidates.append((iou, gt_idx))
        candidates.sort(key=lambda x: x[0], reverse=True)
        prepared.append({
            "image_name": pred["image_name"],
            "score": pred["score"],
            "instance_mask": pred["instance_mask"],
            "candidates": candidates,
        })
    return prepared


def evaluate_ap_for_threshold(prepared_predictions, gt_instances_for_class, total_gt: int, iou_threshold: float):
    if total_gt == 0:
        return np.nan, 0, 0

    matched = {
        image_name: np.zeros(len(instances), dtype=bool)
        for image_name, instances in gt_instances_for_class.items()
    }

    tp = np.zeros(len(prepared_predictions), dtype=np.float64)
    fp = np.zeros(len(prepared_predictions), dtype=np.float64)

    for idx, pred in enumerate(prepared_predictions):
        if not pred["candidates"]:
            fp[idx] = 1.0
            continue

        best_gt_idx = -1
        for iou, gt_idx in pred["candidates"]:
            if iou < iou_threshold:
                break
            if matched[pred["image_name"]][gt_idx]:
                continue
            best_gt_idx = gt_idx
            break

        if best_gt_idx >= 0:
            matched[pred["image_name"]][best_gt_idx] = True
            tp[idx] = 1.0
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    ap = compute_ap_from_pr(recalls, precisions)
    return ap, int(tp.sum()), int(fp.sum())


def evaluate_detection_ap(pred_dir: Path, score_rows, class_id_to_gt_raw: dict, eval_raw_values, image_names, gt_dir: Path):
    gt_instances, total_gt_by_class = build_gt_instances(gt_dir, image_names, eval_raw_values)
    predictions_by_class = load_prediction_instances(pred_dir, score_rows, class_id_to_gt_raw)
    prepared_predictions_by_class = {}
    for raw_value in tqdm(eval_raw_values, desc="Preparing AP matches", leave=False):
        prepared_predictions_by_class[raw_value] = prepare_class_predictions(
            predictions_by_class.get(raw_value, []),
            gt_instances[raw_value],
        )

    per_class_threshold_ap = {raw_value: {} for raw_value in eval_raw_values}
    threshold_mean_ap = {}

    for threshold in tqdm(IOU_THRESHOLDS, desc="Computing AP", leave=False):
        class_aps = []
        for raw_value in eval_raw_values:
            ap, tp_count, fp_count = evaluate_ap_for_threshold(
                prepared_predictions=prepared_predictions_by_class.get(raw_value, []),
                gt_instances_for_class=gt_instances[raw_value],
                total_gt=total_gt_by_class[raw_value],
                iou_threshold=threshold,
            )
            per_class_threshold_ap[raw_value][threshold] = {
                "ap": ap,
                "num_gt": int(total_gt_by_class[raw_value]),
                "num_pred": int(len(predictions_by_class.get(raw_value, []))),
                "tp": tp_count,
                "fp": fp_count,
            }
            if not np.isnan(ap):
                class_aps.append(ap)
        threshold_mean_ap[threshold] = float(np.mean(class_aps)) if class_aps else np.nan

    valid_threshold_aps = [threshold_mean_ap[t] for t in IOU_THRESHOLDS if not np.isnan(threshold_mean_ap[t])]
    ap = float(np.mean(valid_threshold_aps)) if valid_threshold_aps else np.nan
    ap50 = threshold_mean_ap[0.50]
    ap75 = threshold_mean_ap[0.75]

    return {
        "ap": ap,
        "ap50": ap50,
        "ap75": ap75,
        "threshold_mean_ap": threshold_mean_ap,
        "per_class_threshold_ap": per_class_threshold_ap,
    }


def load_score_rows(score_csv: Path):
    with open(score_csv, newline="") as f:
        return list(csv.DictReader(f))


def save_outputs(pred_dir: Path, summary: dict, semantic_eval: dict, detection_eval: dict):
    summary_json = pred_dir / "instance_ap_eval_summary.json"
    summary_csv = pred_dir / "instance_ap_eval_summary.csv"
    per_image_csv = pred_dir / "instance_ap_eval_per_image.csv"
    per_threshold_csv = pred_dir / "instance_ap_eval_per_threshold.csv"

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    summary_df = pd.DataFrame([{
        "AP": summary["AP"] * 100 if not np.isnan(summary["AP"]) else np.nan,
        "AP50": summary["AP50"] * 100 if not np.isnan(summary["AP50"]) else np.nan,
        "AP75": summary["AP75"] * 100 if not np.isnan(summary["AP75"]) else np.nan,
        "mIoU_I": summary["mIoU_I"] * 100 if not np.isnan(summary["mIoU_I"]) else np.nan,
        "mIoU_D": summary["mIoU_D"] * 100 if not np.isnan(summary["mIoU_D"]) else np.nan,
    }])
    summary_df.to_csv(summary_csv, index=False)

    pd.DataFrame(semantic_eval["per_image_rows"]).to_csv(per_image_csv, index=False)

    threshold_rows = []
    for threshold, ap in detection_eval["threshold_mean_ap"].items():
        threshold_rows.append({
            "iou_threshold": threshold,
            "mean_ap": ap * 100 if not np.isnan(ap) else np.nan,
        })
    pd.DataFrame(threshold_rows).to_csv(per_threshold_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate instance AP from predicted instance masks + scores, and semantic mIoU "
            "from pred_dir/semantic_masks against gt_dir semantic masks."
        )
    )
    parser.add_argument("--pred_dir", required=True, help="Prediction directory containing instances/, semantic_masks/, instance_scores.csv")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth semantic mask directory")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    semantic_dir = pred_dir / "semantic_masks"
    score_csv = pred_dir / "instance_scores.csv"

    if not semantic_dir.is_dir():
        raise FileNotFoundError(f"semantic_masks directory not found: {semantic_dir}")
    if not (pred_dir / "instances").is_dir():
        raise FileNotFoundError(f"instances directory not found: {pred_dir / 'instances'}")
    if not score_csv.is_file():
        raise FileNotFoundError(f"instance_scores.csv not found: {score_csv}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory not found: {gt_dir}")

    gt_files_map = {p.name: p for p in gt_dir.iterdir() if is_image_file(p)}
    image_names = sorted([p.name for p in semantic_dir.iterdir() if is_image_file(p) and p.name in gt_files_map])
    if not image_names:
        raise RuntimeError("No overlapping prediction/GT image names found.")

    score_rows = load_score_rows(score_csv)
    mappings = infer_class_mappings(pred_dir, gt_dir, image_names, score_rows)
    eval_raw_values = mappings["gt_fg_values"]

    semantic_eval = evaluate_semantic_miou(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        image_names=image_names,
        pred_raw_to_gt_raw=mappings["pred_raw_to_gt_raw"],
        eval_raw_values=eval_raw_values,
    )
    detection_eval = evaluate_detection_ap(
        pred_dir=pred_dir,
        score_rows=score_rows,
        class_id_to_gt_raw=mappings["class_id_to_gt_raw"],
        eval_raw_values=eval_raw_values,
        image_names=image_names,
        gt_dir=gt_dir,
    )

    summary = {
        "pred_dir": str(pred_dir.resolve()),
        "gt_dir": str(gt_dir.resolve()),
        "matched_images": len(image_names),
        "evaluated_gt_raw_values": eval_raw_values,
        "pred_raw_values": mappings["pred_raw_values"],
        "gt_raw_values": mappings["gt_raw_values"],
        "class_id_to_pred_raw": {str(k): int(v) for k, v in mappings["class_id_to_pred_raw"].items()},
        "class_id_to_gt_raw": {str(k): int(v) for k, v in mappings["class_id_to_gt_raw"].items()},
        "AP": detection_eval["ap"],
        "AP50": detection_eval["ap50"],
        "AP75": detection_eval["ap75"],
        "mIoU_I": semantic_eval["miou_i"],
        "mIoU_D": semantic_eval["miou_d"],
        "per_class_mIoU_D": {str(k): v for k, v in semantic_eval["per_class_miou_d"].items()},
        "AP_by_threshold": {str(k): v for k, v in detection_eval["threshold_mean_ap"].items()},
        "per_class_AP_by_threshold": {
            str(raw): {str(t): v for t, v in threshold_dict.items()}
            for raw, threshold_dict in detection_eval["per_class_threshold_ap"].items()
        },
    }

    save_outputs(pred_dir, summary, semantic_eval, detection_eval)

    print(f"Pred dir: {pred_dir.resolve()}")
    print(f"GT dir:   {gt_dir.resolve()}")
    print(f"Matched images: {len(image_names)}")
    print(f"Evaluated GT classes (non-background): {eval_raw_values}")
    print(f"AP:     {summary['AP'] * 100:.2f}")
    print(f"AP50:   {summary['AP50'] * 100:.2f}")
    print(f"AP75:   {summary['AP75'] * 100:.2f}")
    print(f"mIoU^I: {summary['mIoU_I'] * 100:.2f}")
    print(f"mIoU^D: {summary['mIoU_D'] * 100:.2f}")
    print(f"Saved summary to: {(pred_dir / 'instance_ap_eval_summary.json').resolve()}")


if __name__ == "__main__":
    main()
