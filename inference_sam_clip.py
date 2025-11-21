#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Resize input to 1024x1024 (no ResizeLongestSide)
- ToTensor + ImageNet mean/std normalization
- Forward:
    SAM-CLIP fused image embedding (SAM encoder + CLIP image + CLIP text + FC layers)
    -> prompt_encoder(None) -> mask_decoder(multimask_output=True)
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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm

from models.sam import sam_model_registry
import clip


def evaluate_one(
    image_path: str,
    model,
    fc_sam_to_clip: nn.Module,
    fc_fuse_to_decoder: nn.Module,
    clip_model,
    clip_preprocess,
    clip_text_feat: torch.Tensor,
    device: str = "cuda",
):
    """
    Single-image forward pass keeping ipynb-style preprocessing and
    SAM-CLIP fusion for image embeddings. Reduces grid artifacts by
    upsampling logits (not labels) to original size before argmax.
    Applies light morphological smoothing for binary masks.
    """
    # Load original image
    pil_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_orig.size

    # Preprocessing: resize to 1024x1024 + ImageNet normalization
    preprocess = transforms.Compose([
        transforms.Resize((1024, 1024)),                     # fixed 1024x1024
        transforms.ToTensor(),                               # [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],     # ImageNet mean/std
                             std=[0.229, 0.224, 0.225]),
    ])
    batch = preprocess(pil_orig).unsqueeze(0).to(device)     # [1,3,1024,1024]

    with torch.no_grad():
        # -------- SAM-CLIP fused image embeddings --------
        # 1) SAM image encoder
        sam_img_emb = model.image_encoder(batch)             # [B,256,64,64] typically
        B = sam_img_emb.shape[0]

        # 2) Map SAM feature map to CLIP feature space (512-d)
        sam_feat = fc_sam_to_clip(sam_img_emb.view(B, -1))   # [B,512]

        # 3) CLIP image encoder on the original RGB image
        clip_img = clip_preprocess(pil_orig).unsqueeze(0).to(device)  # [B,3,Hc,Wc]
        clip_feat = clip_model.encode_image(clip_img)                 # [B,512]
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)  # L2 normalize

        # 4) Fuse SAM and CLIP image features
        merged_feat = sam_feat + clip_feat                            # [B,512]

        # 5) Fuse with CLIP text feature via outer product
        #    clip_text_feat: [1,512] -> expand to [B,512]
        text_feat_exp = clip_text_feat.expand(B, -1)                  # [B,512]
        # Outer product: [B,512,1] x [B,1,512] -> [B,512,512]
        dot = torch.bmm(merged_feat.unsqueeze(2), text_feat_exp.unsqueeze(1))
        # Flatten and project back to SAM feature map shape
        fused = fc_fuse_to_decoder(dot.view(B, -1))                   # [B,256*64*64]
        img_emb = fused.view(B, 256, 64, 64)                          # [B,256,64,64]

        # -------- Prompt encoder (no prompts) + mask decoder --------
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
        # Binary model: sigmoid + 0.5 threshold
        prob = torch.sigmoid(logits_full)              # [1,1,H,W]
        pred = (prob[0, 0] > 0.5).to(torch.uint8)      # HxW in {0,1}
    else:
        # Multi-class model: softmax + argmax
        pred = logits_full.softmax(dim=1).argmax(dim=1)[0].to(torch.uint8)  # HxW in {0..C-1}

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
    """Load SAM model using args.json and checkpoint_best.pth from checkpoint_dir."""
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
        description="Batch inference aligned with ipynb-style preprocessing and SAM-CLIP fusion."
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory with args.json and checkpoint_best.pth")
    parser.add_argument("--image_dir", required=True,
                        help="Directory of input images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save predicted masks")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for inference")
    parser.add_argument("--text_prompt", type=str, default="Powdery mildew",
                        help="Text prompt for CLIP encoding, for example 'Powdery mildew'")
    args = parser.parse_args()

    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load SAM backbone (same as original script) ----
    model, model_args = load_model(ckpt_dir, device=device)

    # ---- Load FC layers for SAM-CLIP fusion from checkpoint ----
    ckpt_path = ckpt_dir / 'checkpoint_best.pth'
    ckpt = torch.load(str(ckpt_path), map_location=device)

    # Shapes must match training time: 256*64*64 -> 512, and 512*512 -> 256*64*64
    fc_sam_to_clip = nn.Linear(256 * 64 * 64, 512).to(device)
    fc_fuse_to_decoder = nn.Linear(512 * 512, 256 * 64 * 64).to(device)

    if 'fc_sam_to_clip' not in ckpt or 'fc_fuse_to_decoder' not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'fc_sam_to_clip' or 'fc_fuse_to_decoder'. "
            "Please make sure you are using a SAM-CLIP fine-tuned checkpoint."
        )

    fc_sam_to_clip.load_state_dict(ckpt['fc_sam_to_clip'])
    fc_fuse_to_decoder.load_state_dict(ckpt['fc_fuse_to_decoder'])
    fc_sam_to_clip.eval()
    fc_fuse_to_decoder.eval()

    # ---- Load CLIP model and preprocess (frozen) ----
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    # ---- Precompute CLIP text feature from user prompt ----
    text_tokens = clip.tokenize([args.text_prompt]).to(device)
    with torch.no_grad():
        clip_text_feat = clip_model.encode_text(text_tokens)          # [1,512]
        clip_text_feat = clip_text_feat / clip_text_feat.norm(dim=-1, keepdim=True)

    # Collect valid image files (same as original script)
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    print(f"[INFO] Saving all predicted masks to: {out_dir.resolve()}")

    # Inference with tqdm progress bar
    for img_path in tqdm(image_paths, desc="[INFO] Running inference", ncols=100):
        fname = img_path.stem
        pil_mask = evaluate_one(
            str(img_path),
            model,
            fc_sam_to_clip,
            fc_fuse_to_decoder,
            clip_model,
            clip_preprocess,
            clip_text_feat,
            device=device,
        )

        mask_arr = np.array(pil_mask).astype(np.uint8)
        # For binary models, map {0,1} -> {0,255} for easier visualization
        if getattr(model, "num_classes", None) == 2 or getattr(model_args, "num_cls", None) == 2:
            mask_arr = (mask_arr * 255).astype(np.uint8)

        out_path = out_dir / f"{fname}.png"
        Image.fromarray(mask_arr, mode='L').save(str(out_path))

    print(f"[INFO] All masks saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

