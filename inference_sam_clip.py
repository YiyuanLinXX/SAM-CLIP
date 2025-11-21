#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM-CLIP inference script

- Resize input to 1024x1024 (no ResizeLongestSide)
- ToTensor + ImageNet mean/std normalization
- Forward:
    * SAM image_encoder to get [B, C, H, W]
    * Flatten + fc_sam_to_clip -> 512-d SAM global embedding
    * CLIP image encoder -> 512-d image embedding
    * CLIP text encoder -> 512-d text embedding (user prompt)
    * Fuse in CLIP space: z_fuse = z_sam + z_clip + z_text
    * fc_fuse_to_decoder: 512 -> C, broadcast to [B, C, H, W]
    * Fused image_embeddings = sam_img_emb + fuse_map
    * prompt_encoder(None) -> mask_decoder(multimask_output=True)
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
import clip  # OpenAI CLIP

from models.sam import sam_model_registry


def build_clip_and_text(text_prompt: str, device: str = "cuda"):
    """
    Load frozen CLIP model and preprocess, and compute normalized text embedding.
    """
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    text_tokens = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        clip_text_feat = clip_model.encode_text(text_tokens)  # [1, 512]
        clip_text_feat = clip_text_feat / clip_text_feat.norm(dim=-1, keepdim=True)

    return clip_model, clip_preprocess, clip_text_feat


def evaluate_one(
    image_path: str,
    sam_model,
    fc_sam_to_clip: nn.Module,
    fc_fuse_to_decoder: nn.Module,
    clip_model,
    clip_preprocess,
    clip_text_feat: torch.Tensor,
    device: str = "cuda"
):
    """
    Single-image forward pass for SAM-CLIP:
    - SAM encoder + CLIP image/text fusion in 512-d space
    - Channel-wise modulation back to [B, C, H, W]
    - Mask decoding and light post-processing
    """
    pil_orig = Image.open(image_path).convert('RGB')
    orig_w, orig_h = pil_orig.size

    # Preprocessing for SAM: resize to 1024x1024 + ImageNet normalization
    preprocess_sam = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    batch_sam = preprocess_sam(pil_orig).unsqueeze(0).to(device)  # [1, 3, 1024, 1024]

    # Preprocessing for CLIP: use official preprocess (includes resize+center crop etc.)
    clip_img = clip_preprocess(pil_orig).unsqueeze(0).to(device)  # [1, 3, 224, 224] for ViT-B/32

    with torch.no_grad():
        # === SAM image encoder ===
        sam_img_emb = sam_model.image_encoder(batch_sam)  # [1, C, H, W]
        B, C, H, W = sam_img_emb.shape

        # === Global projection to CLIP space (SAM side) ===
        sam_feat = fc_sam_to_clip(sam_img_emb.view(B, -1))  # [B, 512]

        # === CLIP image encoder ===
        clip_feat = clip_model.encode_image(clip_img)  # [B, 512]
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        # === Fuse SAM global embedding, CLIP image embedding, and text embedding ===
        text_feat_exp = clip_text_feat.expand(B, -1)  # [B, 512]
        merged_feat = sam_feat + clip_feat + text_feat_exp  # [B, 512]

        # === Map fused semantic embedding back to channel space and broadcast ===
        fuse_channel = fc_fuse_to_decoder(merged_feat)  # [B, C]
        fuse_map = fuse_channel.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]

        # === Fused encoder feature for mask decoder ===
        img_emb = sam_img_emb + fuse_map  # [B, C, H, W]

        # === Prompt encoder and mask decoder ===
        sparse_emb, dense_emb = sam_model.prompt_encoder(points=None, boxes=None, masks=None)
        logits_256, _ = sam_model.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
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
    """
    Load SAM backbone and the two MLPs (fc_sam_to_clip, fc_fuse_to_decoder)
    using args.json and checkpoint_best.pth from checkpoint_dir.
    """
    args_json = checkpoint_dir / 'args.json'
    ckpt_path = checkpoint_dir / 'checkpoint_best.pth'
    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found in {checkpoint_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint_best.pth not found in {checkpoint_dir}")

    # Load training args
    with open(args_json, 'r') as f:
        model_args = argparse.Namespace(**json.load(f))

    # Build SAM model with base SAM checkpoint
    sam_model = sam_model_registry[model_args.arch](
        model_args,
        checkpoint=str(model_args.sam_ckpt),  # base SAM checkpoint path stored in args
        num_classes=model_args.num_cls
    ).to(device)

    # Load saved weights
    ckpt = torch.load(ckpt_path, map_location=device)
    sam_model.load_state_dict(ckpt['sam'])

    # Probe encoder output shape to reconstruct the two MLPs
    sam_model.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 1024, 1024), device=device)
        dummy_emb = sam_model.image_encoder(dummy)  # [1, C, H, W]
        C, H, W = dummy_emb.shape[1:]

    # Rebuild the two MLPs with correct shapes
    fc_sam_to_clip = nn.Linear(C * H * W, 512).to(device)
    fc_fuse_to_decoder = nn.Linear(512, C).to(device)

    fc_sam_to_clip.load_state_dict(ckpt['fc_sam_to_clip'])
    fc_fuse_to_decoder.load_state_dict(ckpt['fc_fuse_to_decoder'])

    sam_model.eval()
    fc_sam_to_clip.eval()
    fc_fuse_to_decoder.eval()

    return sam_model, fc_sam_to_clip, fc_fuse_to_decoder, model_args


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for SAM-CLIP (image + text fusion) with ipynb-style preprocessing and binary-mask smoothing."
    )
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Directory with args.json and checkpoint_best.pth (SAM-CLIP training output)")
    parser.add_argument("--image_dir", required=True,
                        help="Directory of input images")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save predicted masks")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for inference")
    parser.add_argument("--text_prompt", default="Powdery mildew",
                        help="Text prompt used for CLIP text embedding (same semantic as training).")
    args = parser.parse_args()

    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM-CLIP components
    sam_model, fc_sam_to_clip, fc_fuse_to_decoder, model_args = load_model(ckpt_dir, device=device)

    # Load CLIP and text embedding
    clip_model, clip_preprocess, clip_text_feat = build_clip_and_text(
        text_prompt=args.text_prompt,
        device=device
    )

    # Collect valid image files
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    print(f"[INFO] Saving all predicted masks to: {out_dir.resolve()}")

    # Inference with tqdm progress bar
    for img_path in tqdm(image_paths, desc="[INFO] Running SAM-CLIP inference", ncols=100):
        fname = img_path.stem
        pil_mask = evaluate_one(
            str(img_path),
            sam_model=sam_model,
            fc_sam_to_clip=fc_sam_to_clip,
            fc_fuse_to_decoder=fc_fuse_to_decoder,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_text_feat=clip_text_feat,
            device=device
        )

        mask_arr = np.array(pil_mask).astype(np.uint8)
        # For binary models, map {0,1} -> {0,255} for easier visualization
        if getattr(sam_model, "num_classes", None) == 2 or getattr(model_args, "num_cls", None) == 2:
            mask_arr = (mask_arr * 255).astype(np.uint8)

        out_path = out_dir / f"{fname}.png"
        Image.fromarray(mask_arr, mode='L').save(str(out_path))

    print(f"[INFO] All masks saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
