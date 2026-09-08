#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference for semantic segmentation with extra outputs for instance AP:
- Keeps the original semantic-mask PNG outputs in output_dir
- Saves per-image probability maps to output_dir/prob_maps/*.npy
- Splits predicted semantic masks into connected-component instances
- Saves instance masks to output_dir/instances/*.png
- Saves per-instance confidence scores to output_dir/instance_scores.csv

The CLI is intentionally identical to inference_sam_clip.py for drop-in usage.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from tqdm import tqdm

from models.sam import sam_model_registry


def build_preprocess(normalize_type="sam"):
    steps = [transforms.Resize((1024, 1024)), transforms.ToTensor()]
    if normalize_type == "sam":
        steps.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    elif normalize_type == "medsam":
        steps.append(
            transforms.Lambda(
                lambda tensor: (tensor - tensor.min())
                / (tensor.max() - tensor.min()).clamp_min(1e-8)
            )
        )
    else:
        raise ValueError(f"Unsupported normalize_type: {normalize_type}")
    return transforms.Compose(steps)


def smooth_binary_mask(pred_np: np.ndarray) -> np.ndarray:
    uniq_vals = np.unique(pred_np)
    if uniq_vals.size <= 2 and uniq_vals.max() <= 1:
        kernel = np.ones((3, 3), np.uint8)
        pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
        pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_CLOSE, kernel)
    return pred_np


def logits_to_prediction_and_probs(logits_full: torch.Tensor):
    """Return predicted labels and full per-class probability maps."""
    if logits_full.shape[1] == 1:
        fg_prob = torch.sigmoid(logits_full)[0, 0]
        bg_prob = 1.0 - fg_prob
        prob_maps = torch.stack([bg_prob, fg_prob], dim=0)
        pred = (fg_prob > 0.5).to(torch.uint8)
    else:
        prob_maps = logits_full.softmax(dim=1)[0]
        pred = prob_maps.argmax(dim=0).to(torch.uint8)
    return pred, prob_maps


def evaluate_one(image_path: str, model, preprocess, device: str = "cuda"):
    pil_orig = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    orig_w, orig_h = pil_orig.size
    batch = preprocess(pil_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = model.image_encoder(batch)
        sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=None, masks=None)
        logits_256, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )

    logits_full = F.interpolate(
        logits_256, size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    pred, prob_maps = logits_to_prediction_and_probs(logits_full)

    pred_np = pred.cpu().numpy().astype(np.uint8)
    pred_np = smooth_binary_mask(pred_np)
    prob_maps_np = prob_maps.cpu().numpy().astype(np.float32)

    return pred_np, prob_maps_np


def load_model(checkpoint_dir: Path, device: str):
    args_json = checkpoint_dir / "args.json"
    ckpt_path = checkpoint_dir / "checkpoint_best.pth"
    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found in {checkpoint_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint_best.pth not found in {checkpoint_dir}")

    with open(args_json, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    model_args.if_split_encoder_gpus = False
    model_args.devices = [0, 1]

    model = sam_model_registry[model_args.arch](
        model_args,
        checkpoint=str(ckpt_path),
        num_classes=model_args.num_cls,
    )
    return model.to(device).eval(), model_args


def save_semantic_mask(pred_np: np.ndarray, out_path: Path, model, model_args):
    mask_arr = pred_np.copy()
    if getattr(model, "num_classes", None) == 2 or getattr(model_args, "num_cls", None) == 2:
        mask_arr = (mask_arr * 255).astype(np.uint8)
    Image.fromarray(mask_arr, mode="L").save(str(out_path))


def extract_and_save_instances(
    pred_np: np.ndarray,
    prob_maps_np: np.ndarray,
    image_stem: str,
    instance_dir: Path,
):
    records = []
    class_ids = [int(v) for v in np.unique(pred_np) if int(v) != 0]

    for class_id in class_ids:
        binary_mask = (pred_np == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        instance_idx = 0
        for comp_id in range(1, num_labels):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue

            instance_mask = labels == comp_id
            score = float(prob_maps_np[class_id][instance_mask].mean())
            out_name = f"{image_stem}_{class_id}_{instance_idx}.png"
            out_path = instance_dir / out_name

            Image.fromarray(instance_mask.astype(np.uint8) * 255, mode="L").save(str(out_path))

            records.append({
                "image_name": f"{image_stem}.png",
                "instance_mask": out_name,
                "class_id": class_id,
                "instance_id": instance_idx,
                "score": score,
                "area": area,
            })
            instance_idx += 1

    return records


def write_score_csv(csv_path: Path, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_name", "instance_mask", "class_id", "instance_id", "score", "area"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic segmentation inference with saved probability maps and per-instance scores."
    )
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with args.json and checkpoint_best.pth")
    parser.add_argument("--image_dir", required=True, help="Directory of input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save predicted masks")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no NVIDIA GPU is visible; run `sam-clip gpu-check`.")
    ckpt_dir = Path(args.checkpoint_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    semantic_dir = out_dir / "semantic_masks"
    prob_dir = out_dir / "prob_maps"
    instance_dir = out_dir / "instances"
    score_csv = out_dir / "instance_scores.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    prob_dir.mkdir(parents=True, exist_ok=True)
    instance_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = load_model(ckpt_dir, device=device)
    preprocess = build_preprocess(getattr(model_args, "normalize_type", "sam"))

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not image_paths:
        raise RuntimeError(f"No supported images found in {img_dir}")

    print(f"[INFO] Saving semantic masks to: {semantic_dir.resolve()}")
    print(f"[INFO] Saving probability maps to: {prob_dir.resolve()}")
    print(f"[INFO] Saving instance masks to: {instance_dir.resolve()}")

    all_rows = []
    for img_path in tqdm(image_paths, desc="[INFO] Running inference", ncols=100):
        image_stem = img_path.stem
        pred_np, prob_maps_np = evaluate_one(str(img_path), model, preprocess, device=device)

        save_semantic_mask(pred_np, semantic_dir / f"{image_stem}.png", model, model_args)
        np.save(prob_dir / f"{image_stem}.npy", prob_maps_np)

        rows = extract_and_save_instances(pred_np, prob_maps_np, image_stem, instance_dir)
        all_rows.extend(rows)

    write_score_csv(score_csv, all_rows)

    print(f"[INFO] All semantic masks saved to: {semantic_dir.resolve()}")
    print(f"[INFO] All probability maps saved to: {prob_dir.resolve()}")
    print(f"[INFO] All instance masks saved to: {instance_dir.resolve()}")
    print(f"[INFO] Instance scores saved to: {score_csv.resolve()}")


if __name__ == "__main__":
    main()
