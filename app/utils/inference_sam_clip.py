#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_noprompt_ipynb_style.py

Batch inference that mirrors a typical official 2D ipynb style:
- Resize input to 1024x1024 (no ResizeLongestSide)
- ToTensor + ImageNet mean/std normalization
- Forward: image_encoder -> prompt_encoder(None) -> mask_decoder(multimask_output=True)
- Upsample logits (not labels) back to the original size, then argmax
- Lightweight post-processing smoothing (binary masks only): 3x3 morphological open + close
- Save per-image predicted masks
- tqdm progress bar; print output folder path at start and end only
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm

from models.sam import sam_model_registry


def evaluate_one(image_path: str, model, device: str = "cuda"):
    """Single-image forward pass keeping ipynb-style preprocessing,
    reducing grid artifacts by upsampling logits (not labels) to original size
    before argmax. Applies light morphological smoothing for binary masks."""
    pil_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_orig.size

    # Preprocessing: resize to 1024x1024 + ImageNet normalization
    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)),                     # fixed 1024×1024
        transforms.ToTensor(),                               # [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet mean/std
                             std=[0.229, 0.224, 0.225]),
    ])
    batch = preprocess(pil_orig).unsqueeze(0).to(device)

    with torch.no_grad():
        img_emb = model.image_encoder(batch)
        sparse_emb, dense_emb = model.prompt_encoder(points=None, boxes=None, masks=None)
        logits_256, _ = model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True
        )

    # Upsample logits (not labels) to original image size
    logits_full = F.interpolate(
        logits_256, size=(orig_h, orig_w), mode='bilinear', align_corners=False
    )

    # Convert logits to labels
    if logits_full.shape[1] == 1:
        prob = torch.sigmoid(logits_full)              # [1,1,H,W]
        pred = (prob[0, 0] > 0.5).to(torch.uint8)      # H×W in {0,1}
    else:
        pred = logits_full.softmax(dim=1).argmax(dim=1)[0].to(torch.uint8)  # H×W in {0..C-1}

    # Post-processing smoothing (binary masks only)
    pred_np = pred.cpu().numpy()
    uniq_vals = np.unique(pred_np)
    if uniq_vals.size <= 2 and uniq_vals.max() <= 1:
        kernel = np.ones((3, 3), np.uint8)
        pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
        pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_CLOSE, kernel)

    pil_mask = Image.fromarray(pred_np.astype(np.uint8), mode='L')
    return pil_mask


def load_model(checkpoint_dir: Path, device: str):
    """Load model using args.json and checkpoint_best.pth from checkpoint_dir."""
    args_json = checkpoint_dir / 'args.json'
    ckpt_path = checkpoint_dir / 'checkpoint_best.pth'
    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found in {checkpoint_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint_best.pth not found in {checkpoint_dir}")

    with open(args_json, 'r') as f:
        model_args = argparse.Namespace(**json.load(f))
    model = sam_model_registry[model_args.arch](
        model_args,
        checkpoint=str(ckpt_path),
        num_classes=model_args.num_cls
    )
    return model.to(device).eval(), model_args


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference aligned with ipynb-style preprocessing and binary-mask smoothing."
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory with args.json and checkpoint_best.pth")
    parser.add_argument("--image_dir", required=True,
                        help="Directory of input images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save predicted masks")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for inference")
    args = parser.parse_args()

    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_args = load_model(ckpt_dir, device=device)

    # Collect valid image files
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    print(f"[INFO] Saving all predicted masks to: {out_dir.resolve()}")

    # Inference with tqdm progress bar
    for img_path in tqdm(image_paths, desc="[INFO] Running inference", ncols=100):
        fname = img_path.stem
        pil_mask = evaluate_one(str(img_path), model, device=device)

        mask_arr = np.array(pil_mask).astype(np.uint8)
        # For binary models, map {0,1} -> {0,255} for easier visualization
        if getattr(model, "num_classes", None) == 2 or getattr(model_args, "num_cls", None) == 2:
            mask_arr = (mask_arr * 255).astype(np.uint8)

        out_path = out_dir / f"{fname}.png"
        Image.fromarray(mask_arr, mode='L').save(str(out_path))

    print(f"[INFO] All masks saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
