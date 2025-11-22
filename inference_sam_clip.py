#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference script for SAM-CLIP model with image-image fusion and text-modulated multimodal embedding.

Given a fine-tuned checkpoint directory containing:
  - args.json
  - checkpoint_best.pth  (with keys: "sam", "fc_sam_to_clip", "semantic_decoder")

This script:
  1) Loads the SAM backbone and fusion heads (ProjectionHead + SemanticDecoder).
  2) Loads a frozen CLIP ViT-B/32 model.
  3) For each image in the input directory:
       - Runs SAM image encoder to obtain dense features.
       - Applies a spatial bottleneck (8×8) and projection head to obtain a 512-d SAM embedding.
       - Runs CLIP image encoder to obtain a 512-d image embedding.
       - Fuses SAM and CLIP image embeddings:
           image_fuse = e_SAM + e_CLIP_img
       - Computes CLIP text embedding from the text prompt and expands to batch.
       - Applies element-wise multiplication:
           multi_modal = image_fuse * e_text
       - Decodes multi_modal into dense image embeddings (C×H×W).
       - Runs SAM mask decoder to obtain segmentation logits.
       - Upsamples logits to original image size and converts to a label mask.
  4) Saves the predicted mask as a single-channel PNG for each input image.
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
from tqdm import tqdm
import cv2
import clip

from models.sam import sam_model_registry


# -------------------------------------------------------------------------
# Fusion heads (must match training script definitions)
# -------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    Projection head that maps spatially pooled SAM features into the 512-d CLIP semantic space.

    Input:
        x: [B, in_dim], where in_dim = C * bottleneck_h * bottleneck_w
    Output:
        [B, 512]
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class SemanticDecoder(nn.Module):
    """
    Lightweight decoder that maps a fused 512-d multimodal embedding to dense image embeddings.

    Input:
        x: [B, in_dim] where in_dim = 512
    Output:
        [B, out_dim] where out_dim = C * H * W
    """

    def __init__(self, in_dim: int, latent_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------

def load_model(checkpoint_dir: Path, device: str = "cuda", text_prompt_override: str | None = None):
    """
    Load SAM backbone, projection head, semantic decoder, and CLIP components
    from a checkpoint directory containing args.json and checkpoint_best.pth.

    Optionally override the text prompt via text_prompt_override.
    """
    args_json = checkpoint_dir / "args.json"
    ckpt_path = checkpoint_dir / "checkpoint_best.pth"

    if not args_json.exists():
        raise FileNotFoundError(f"args.json not found in {checkpoint_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint_best.pth not found in {checkpoint_dir}")

    # Load training arguments
    with open(args_json, "r") as f:
        args_dict = json.load(f)

    # Simple namespace-like object
    class ArgsObj:
        pass

    model_args = ArgsObj()
    for k, v in args_dict.items():
        setattr(model_args, k, v)

    # Ensure text_prompt exists
    if text_prompt_override is not None:
        text_prompt = text_prompt_override
    else:
        text_prompt = getattr(model_args, "text_prompt", "Powdery mildew")

    # Device
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Load CLIP (frozen)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()

    # Text embedding
    text_tokens = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        clip_text_feat = clip_model.encode_text(text_tokens)  # [1, 512]
        clip_text_feat = clip_text_feat / clip_text_feat.norm(dim=-1, keepdim=True)

    # Instantiate SAM backbone
    # Use the original SAM checkpoint path from args (e.g., "sam_vit_b_01ec64.pth")
    sam_ckpt_path = getattr(model_args, "sam_ckpt", None)
    if sam_ckpt_path is None:
        raise ValueError("sam_ckpt not found in args.json")

    sam = sam_model_registry[model_args.arch](
        model_args,
        checkpoint=str(sam_ckpt_path),
        num_classes=model_args.num_cls,
    )
    sam.to(device)
    sam.eval()

    # Probe encoder output shape (C, H, W)
    image_size = getattr(model_args, "image_size", 1024)
    with torch.no_grad():
        dummy = torch.zeros((1, 3, image_size, image_size), device=device)
        dummy_emb = sam.image_encoder(dummy)  # [1, C, H, W]
        C, H, W = dummy_emb.shape[1:]

    # Define spatial bottleneck
    bottleneck_grid = 8
    bottleneck_h, bottleneck_w = bottleneck_grid, bottleneck_grid
    sam_bottleneck_dim = C * bottleneck_h * bottleneck_w

    # Instantiate fusion heads
    fc_sam_to_clip = ProjectionHead(
        in_dim=sam_bottleneck_dim,
        hidden_dim=512,
        out_dim=512,
    ).to(device)

    latent_dim = 32
    semantic_decoder = SemanticDecoder(
        in_dim=512,
        latent_dim=latent_dim,
        out_dim=C * H * W,
        hidden_dim=256,
    ).to(device)

    # Load checkpoint weights
    checkpoint = torch.load(ckpt_path, map_location=device)
    sam.load_state_dict(checkpoint["sam"], strict=True)
    fc_sam_to_clip.load_state_dict(checkpoint["fc_sam_to_clip"], strict=True)
    semantic_decoder.load_state_dict(checkpoint["semantic_decoder"], strict=True)

    sam.eval()
    fc_sam_to_clip.eval()
    semantic_decoder.eval()

    return (
        sam,
        fc_sam_to_clip,
        semantic_decoder,
        clip_model,
        clip_preprocess,
        clip_text_feat,
        model_args,
        (C, H, W, bottleneck_h, bottleneck_w),
    )


# -------------------------------------------------------------------------
# Single-image inference
# -------------------------------------------------------------------------

def evaluate_one(
    image_path: str,
    sam,
    fc_sam_to_clip,
    semantic_decoder,
    clip_model,
    clip_preprocess,
    clip_text_feat,
    dims,
    model_args,
    device: str = "cuda",
):
    """
    Run inference on a single image using the SAM-CLIP fusion pipeline.

    Steps:
      1) Preprocess image for SAM (resize + ImageNet normalization).
      2) Run SAM image encoder to obtain dense features [C, H, W].
      3) Apply 8×8 spatial bottleneck and projection head to obtain 512-d SAM embedding.
      4) Preprocess image for CLIP and obtain 512-d CLIP image embedding.
      5) Fuse SAM and CLIP image embeddings:
           image_fuse = e_SAM + e_CLIP_img
      6) Expand CLIP text embedding to batch and apply element-wise multiplication:
           multi_modal = image_fuse * e_text
      7) Decode multi_modal into dense image embeddings (C×H×W).
      8) Run SAM mask decoder to obtain logits.
      9) Upsample logits to original image size, then convert to label mask.
     10) For binary segmentation, optionally map {0,1} to {0,255}.
    """
    C, H_enc, W_enc, bottleneck_h, bottleneck_w = dims

    # Load original image
    pil_orig = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_orig.size

    # Preprocessing for SAM encoder
    image_size = getattr(model_args, "image_size", 1024)
    preprocess_sam = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],    # ImageNet mean/std
            std=[0.229, 0.224, 0.225],
        ),
    ])
    batch_sam = preprocess_sam(pil_orig).unsqueeze(0).to(device)  # [1, 3, image_size, image_size]

    # Preprocessing for CLIP encoder
    clip_img = clip_preprocess(pil_orig).unsqueeze(0).to(device)  # [1, 3, clip_size, clip_size]

    with torch.no_grad():
        # SAM image encoder
        sam_img_emb = sam.image_encoder(batch_sam)  # [1, C, H_enc, W_enc]
        B, C_enc, H_e, W_e = sam_img_emb.shape
        assert C_enc == C and H_e == H_enc and W_e == W_enc, "Encoder output shape mismatch."

        # Spatial bottleneck (8×8) and projection head: SAM -> 512-d
        sam_reduced = F.adaptive_avg_pool2d(sam_img_emb, (bottleneck_h, bottleneck_w))  # [1, C, bh, bw]
        sam_flat = sam_reduced.view(B, C * bottleneck_h * bottleneck_w)                 # [1, C*bh*bw]
        sam_feat = fc_sam_to_clip(sam_flat)                                             # [1, 512]

        # CLIP image feature
        clip_feat = clip_model.encode_image(clip_img)                                   # [1, 512]
        clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

        # CLIP text feature (precomputed, expand to batch)
        text_feat_exp = clip_text_feat.expand(B, -1)                                    # [1, 512]

        # Image-image fusion
        image_fuse = sam_feat + clip_feat                                               # [1, 512]

        # Text-modulated multimodal embedding
        multi_modal = image_fuse * text_feat_exp                                        # [1, 512]

        # Decode multimodal embedding into dense image embeddings
        fused_flat = semantic_decoder(multi_modal)                                      # [1, C*H_enc*W_enc]
        img_emb = fused_flat.view(B, C, H_enc, W_enc)                                   # [1, C, H_enc, W_enc]

        # SAM prompt encoder and mask decoder
        sparse_emb, dense_emb = sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        logits_256, _ = sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=True,
        )  # [1, num_cls, out_size, out_size]

    # Upsample logits to original image size
    logits_full = F.interpolate(
        logits_256,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )  # [1, num_cls, H_orig, W_orig]

    # Convert logits to label mask
    if logits_full.shape[1] == 1:
        # Single-channel logits: treat as binary mask with sigmoid
        prob = torch.sigmoid(logits_full)              # [1, 1, H, W]
        pred = (prob[0, 0] > 0.5).to(torch.uint8)      # [H, W] in {0,1}
    else:
        # Multi-channel logits: use softmax + argmax
        pred = logits_full.softmax(dim=1).argmax(dim=1)[0].to(torch.uint8)  # [H, W] in {0..C-1}

    pred_np = pred.cpu().numpy()

    # Optional lightweight morphological smoothing for near-binary masks
    uniq_vals = np.unique(pred_np)
    if uniq_vals.size <= 2 and uniq_vals.max() <= 1:
        kernel = np.ones((3, 3), np.uint8)
        pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_OPEN, kernel)
        pred_np = cv2.morphologyEx(pred_np, cv2.MORPH_CLOSE, kernel)

    # For binary models (num_cls == 2), map {0,1} -> {0,255} for visualization
    num_cls = getattr(model_args, "num_cls", None)
    if num_cls == 2:
        pred_np = (pred_np * 255).astype(np.uint8)
        pil_mask = Image.fromarray(pred_np, mode="L")
    else:
        # Keep class indices as uint8 label mask
        pil_mask = Image.fromarray(pred_np.astype(np.uint8), mode="L")

    return pil_mask


# -------------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference for SAM-CLIP model with image-image fusion and text-modulated multimodal embeddings."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory containing args.json and checkpoint_best.pth",
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory of input images",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save predicted masks",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    parser.add_argument(
        "--text_prompt",
        default=None,
        help="Optional text prompt to override the one in args.json (for CLIP text encoder).",
    )

    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    img_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model components
    (
        sam,
        fc_sam_to_clip,
        semantic_decoder,
        clip_model,
        clip_preprocess,
        clip_text_feat,
        model_args,
        dims,
    ) = load_model(
        ckpt_dir,
        device=args.device,
        text_prompt_override=args.text_prompt,
    )

    # Collect valid image files
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    image_paths = [p for p in sorted(img_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    if not image_paths:
        print(f"[WARN] No images found in {img_dir}")
        return

    print(f"[INFO] Using text prompt: \"{getattr(model_args, 'text_prompt', 'Powdery mildew') if args.text_prompt is None else args.text_prompt}\"")
    print(f"[INFO] Saving all predicted masks to: {out_dir.resolve()}")

    # Inference with progress bar
    for img_path in tqdm(image_paths, desc="[INFO] Running inference", ncols=100):
        fname = img_path.stem
        pil_mask = evaluate_one(
            str(img_path),
            sam,
            fc_sam_to_clip,
            semantic_decoder,
            clip_model,
            clip_preprocess,
            clip_text_feat,
            dims,
            model_args,
            device=args.device,
        )

        mask_arr = np.array(pil_mask).astype(np.uint8)
        out_path = out_dir / f"{fname}.png"
        Image.fromarray(mask_arr, mode="L").save(str(out_path))

    print(f"[INFO] All masks saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
