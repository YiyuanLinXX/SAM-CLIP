#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAM-CLIP training script with image-image fusion and text-modulated multimodal embedding.

Overview
--------
Given an input image, the model:
1) Extracts dense visual features using the SAM image encoder.
2) Applies a spatial bottleneck (e.g., 8×8) and a projection head to obtain a 512-d semantic
   embedding aligned with the CLIP space.
3) Extracts a 512-d CLIP image embedding and a 512-d CLIP text embedding (from a user-defined
   text prompt).
4) Fuses SAM and CLIP image features in the shared semantic space:
      image_fuse = e_SAM + e_CLIP_img
5) Applies element-wise multiplication with the text embedding to obtain a multimodal embedding:
      multi_modal = image_fuse * e_text
6) Uses a lightweight MLP decoder to transform the multimodal embedding into dense image
   embeddings (C×H×W), which are then fed into the original SAM mask decoder for segmentation.
"""

import os
import json
from pathlib import Path

import torch
import clip
import monai
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms

from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
from utils.dataset import Public_dataset
from utils.dsc import dice_coeff_multi_class
import cfg


# Parse training arguments
args = cfg.parse_args()

# Ensure text_prompt exists
if not hasattr(args, "text_prompt") or args.text_prompt is None:
    args.text_prompt = "Powdery mildew"

# Select device
device = "cuda" if torch.cuda.is_available() and args.gpu else "cpu"

# Load and freeze CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
for p in clip_model.parameters():
    p.requires_grad = False
clip_model.eval()

# Precompute CLIP text embedding from user-defined text prompt
text_tokens = clip.tokenize([args.text_prompt]).to(device)
with torch.no_grad():
    clip_text_feat = clip_model.encode_text(text_tokens)  # [1, 512]
    clip_text_feat = clip_text_feat / clip_text_feat.norm(dim=-1, keepdim=True)


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


def train_model(trainloader, valloader, dir_checkpoint: str, epochs: int):
    # Initialize base learning rate (with optional warmup)
    if args.if_warmup:
        base_lr = args.lr / float(args.warmup_period)
    else:
        base_lr = args.lr

    # Load SAM backbone
    sam = sam_model_registry[args.arch](
        args,
        checkpoint=os.path.join(args.sam_ckpt),
        num_classes=args.num_cls,
    )

    # Apply fine-tuning strategy
    if args.finetune_type == "adapter":
        # Only update adapter parameters
        for name, param in sam.named_parameters():
            if "Adapter" not in name:
                param.requires_grad = False
    elif args.finetune_type == "vanilla" and not args.if_update_encoder:
        # Freeze image encoder, update mask decoder and heads only
        for _, param in sam.image_encoder.named_parameters():
            param.requires_grad = False
    elif args.finetune_type == "lora":
        # Wrap SAM with LoRA modules
        sam = LoRA_Sam(args, sam, r=4).sam

    sam.to(device)

    # Probe encoder output shape (C, H, W) using a dummy input
    sam.eval()
    with torch.no_grad():
        dummy = torch.zeros((1, 3, args.image_size, args.image_size)).to(device)
        dummy_emb = sam.image_encoder(dummy)  # [1, C, H, W]
        C, H, W = dummy_emb.shape[1:]

    # Spatial bottleneck resolution for SAM features (e.g., 8×8)
    bottleneck_grid = 8
    bottleneck_h, bottleneck_w = bottleneck_grid, bottleneck_grid
    sam_bottleneck_dim = C * bottleneck_h * bottleneck_w

    # Projection head: SAM bottleneck features -> 512-d CLIP semantic space
    fc_sam_to_clip = ProjectionHead(
        in_dim=sam_bottleneck_dim,
        hidden_dim=512,
        out_dim=512,
    ).to(device)

    # Semantic decoder: multimodal 512-d embedding -> C * H * W
    latent_dim = 32  # small latent dimension to keep parameter count manageable
    semantic_decoder = SemanticDecoder(
        in_dim=512,
        latent_dim=latent_dim,
        out_dim=C * H * W,
        hidden_dim=256,
    ).to(device)

    # Collect parameters of SAM and fusion modules
    fusion_params = list(fc_sam_to_clip.parameters()) + list(semantic_decoder.parameters())
    optimizer = optim.AdamW(
        list(sam.parameters()) + fusion_params,
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.1,
        amsgrad=False,
    )
    optimizer.zero_grad()

    # Optional step scheduler (learning rate is mainly controlled by custom warmup/decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Loss functions: Dice + Cross-Entropy
    criterion_dice = monai.losses.DiceLoss(
        sigmoid=True,
        squared_pred=True,
        to_onehot_y=True,
        reduction="mean",
    )
    criterion_ce = nn.CrossEntropyLoss()

    # Training bookkeeping
    iter_num = 0
    max_iterations = epochs * len(trainloader)
    writer = SummaryWriter(os.path.join(dir_checkpoint, "log"))

    best_val_dice = 0.0
    last_update_epoch = 0

    pbar = tqdm(range(epochs))
    for epoch in pbar:
        sam.train()
        fc_sam_to_clip.train()
        semantic_decoder.train()

        train_loss = 0.0

        for i, data in enumerate(tqdm(trainloader)):
            imgs = data["image"].to(device)
            # Resize masks to match decoder output size
            msks = torchvision.transforms.Resize(
                (args.out_size, args.out_size)
            )(data["mask"]).to(device)

            # SAM image encoder
            sam_img_emb = sam.image_encoder(imgs)  # [B, C, H, W]
            B, C_enc, H_enc, W_enc = sam_img_emb.shape
            assert C_enc == C and H_enc == H and W_enc == W, "Unexpected encoder output shape."

            # Spatial bottleneck: adaptive pooling to fixed grid (bottleneck_h × bottleneck_w)
            sam_reduced = F.adaptive_avg_pool2d(
                sam_img_emb, (bottleneck_h, bottleneck_w)
            )  # [B, C, bh, bw]
            sam_flat = sam_reduced.view(B, sam_bottleneck_dim)  # [B, C*bh*bw]

            # Projection head: SAM -> CLIP semantic space
            sam_feat = fc_sam_to_clip(sam_flat)  # [B, 512]

            # CLIP image embedding
            imgs_pil = [transforms.ToPILImage()(img.cpu()) for img in imgs]
            clip_imgs = torch.stack([clip_preprocess(pil) for pil in imgs_pil]).to(device)
            with torch.no_grad():
                clip_feat = clip_model.encode_image(clip_imgs)  # [B, 512]
                clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

            # CLIP text embedding (precomputed, only expand along batch)
            text_feat_exp = clip_text_feat.expand(B, -1)  # [B, 512]

            # Step 1: image-image fusion in CLIP space
            image_fuse = sam_feat + clip_feat  # [B, 512]

            # Step 2: text-modulated multimodal embedding
            multi_modal = image_fuse * text_feat_exp  # [B, 512]

            # Decode multimodal embedding into dense image embeddings
            fused_flat = semantic_decoder(multi_modal)  # [B, C*H*W]
            img_emb = fused_flat.view(B, C, H, W)       # [B, C, H, W]

            # SAM prompt encoder and mask decoder
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            # Segmentation loss
            loss_dice = criterion_dice(pred, msks.float())
            loss_ce = criterion_ce(pred, torch.squeeze(msks.long(), 1))
            loss = loss_dice + loss_ce

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Learning rate scheduling: warmup + polynomial decay (optional)
            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * float(iter_num + 1) / float(args.warmup_period)
            elif args.if_warmup:
                shift_iter = iter_num - args.warmup_period
                lr_ = args.lr * (1.0 - float(shift_iter) / float(max_iterations)) ** 0.9
            else:
                lr_ = args.lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            train_loss += loss.item()
            iter_num += 1

            # Logging
            writer.add_scalar("train/lr", lr_, iter_num)
            writer.add_scalar("train/total_loss", loss.item(), iter_num)
            writer.add_scalar("train/loss_ce", loss_ce.item(), iter_num)
            writer.add_scalar("train/loss_dice", loss_dice.item(), iter_num)

        train_loss /= float(i + 1)
        pbar.set_description(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # Validation every 2 epochs
        if epoch % 2 == 0:
            sam.eval()
            fc_sam_to_clip.eval()
            semantic_decoder.eval()

            val_loss = 0.0
            val_dice = 0.0

            with torch.no_grad():
                for i, data in enumerate(tqdm(valloader)):
                    imgs = data["image"].to(device)
                    msks = torchvision.transforms.Resize(
                        (args.out_size, args.out_size)
                    )(data["mask"]).to(device)

                    sam_img_emb = sam.image_encoder(imgs)  # [B, C, H, W]
                    B, C_enc, H_enc, W_enc = sam_img_emb.shape
                    assert C_enc == C and H_enc == H and W_enc == W, "Unexpected encoder output shape."

                    sam_reduced = F.adaptive_avg_pool2d(
                        sam_img_emb, (bottleneck_h, bottleneck_w)
                    )  # [B, C, bh, bw]
                    sam_flat = sam_reduced.view(B, sam_bottleneck_dim)
                    sam_feat = fc_sam_to_clip(sam_flat)  # [B, 512]

                    imgs_pil = [transforms.ToPILImage()(img.cpu()) for img in imgs]
                    clip_imgs = torch.stack([clip_preprocess(pil) for pil in imgs_pil]).to(device)
                    clip_feat = clip_model.encode_image(clip_imgs)  # [B, 512]
                    clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

                    text_feat_exp = clip_text_feat.expand(B, -1)  # [B, 512]

                    image_fuse = sam_feat + clip_feat            # [B, 512]
                    multi_modal = image_fuse * text_feat_exp     # [B, 512]

                    fused_flat = semantic_decoder(multi_modal)   # [B, C*H*W]
                    img_emb = fused_flat.view(B, C, H, W)        # [B, C, H, W]

                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None,
                    )
                    pred, _ = sam.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )

                    loss = criterion_dice(pred, msks.float()) + criterion_ce(
                        pred, torch.squeeze(msks.long(), 1)
                    )
                    val_loss += loss.item()

                    dice_batch = dice_coeff_multi_class(
                        pred.argmax(1).cpu(),
                        torch.squeeze(msks.long(), 1).cpu(),
                        args.num_cls,
                    )
                    val_dice += dice_batch

                val_loss /= float(i + 1)
                val_dice /= float(i + 1)

                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/dice", val_dice, epoch)

                print(f"Eval Epoch {epoch} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")

                # Save best checkpoint
                if val_dice > best_val_dice:
                    best_val_dice = val_dice
                    last_update_epoch = epoch
                    print(f"New best Dice: {val_dice:.4f}")
                    torch.save(
                        {
                            "sam": sam.state_dict(),
                            "fc_sam_to_clip": fc_sam_to_clip.state_dict(),
                            "semantic_decoder": semantic_decoder.state_dict(),
                        },
                        os.path.join(dir_checkpoint, "checkpoint_best.pth"),
                    )
                elif epoch - last_update_epoch > 200:
                    print("Early stopping: no improvement in 200 epochs.")
                    break

    writer.close()


if __name__ == "__main__":
    dataset_name = args.dataset_name
    print(f"Training on dataset: {dataset_name}")
    print(f"Using text prompt for CLIP: \"{args.text_prompt}\"")
    print(f"Checkpoint directory: {args.dir_checkpoint}")

    train_img_list = args.train_img_list
    val_img_list = args.val_img_list

    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.dir_checkpoint, "args.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)

    train_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        train_img_list,
        phase="train",
        targets=[args.targets],
        normalize_type="sam",
        if_prompt=False,
    )
    val_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        val_img_list,
        phase="val",
        targets=[args.targets],
        normalize_type="sam",
        if_prompt=False,
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.w,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=args.b,
        shuffle=False,
        num_workers=args.w,
    )

    train_model(trainloader, valloader, args.dir_checkpoint, args.epochs)
