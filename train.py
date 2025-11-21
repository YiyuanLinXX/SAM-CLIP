# === SAM-CLIP Enhanced Training Script with Lightweight Fusion ===

import torch
import clip
import os
import json
import cv2
import copy
import monai
import numpy as np
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.nn.functional import one_hot

from models.sam import sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from models.sam_LoRa import LoRA_Sam
from utils.dataset import Public_dataset
from utils.losses import DiceLoss
from utils.dsc import dice_coeff_multi_class
import cfg

# Parse training arguments from configuration
args = cfg.parse_args()

# Ensure text prompt is available; allow override from cfg if already defined
if not hasattr(args, "text_prompt"):
    # Default text prompt if not provided in cfg
    args.text_prompt = "Powdery mildew"

# === Load and freeze CLIP model ===
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
for p in clip_model.parameters():
    p.requires_grad = False
clip_model.eval()

# === Precompute fixed text embedding for class label (from user provided text_prompt) ===
text_tokens = clip.tokenize([args.text_prompt]).to("cuda")
with torch.no_grad():
    clip_text_feat = clip_model.encode_text(text_tokens)
    clip_text_feat = clip_text_feat / clip_text_feat.norm(dim=-1, keepdim=True)  # [1, 512]


def train_model(trainloader, valloader, dir_checkpoint, epochs):
    # === Initialize learning rate based on warmup strategy ===
    if args.if_warmup:
        b_lr = args.lr / args.warmup_period
    else:
        b_lr = args.lr

    # === Load SAM backbone model ===
    sam = sam_model_registry[args.arch](args, checkpoint=os.path.join(args.sam_ckpt), num_classes=args.num_cls)

    # === Apply parameter freezing strategy ===
    if args.finetune_type == 'adapter':
        for n, value in sam.named_parameters():
            if "Adapter" not in n:
                value.requires_grad = False
    elif args.finetune_type == 'vanilla' and not args.if_update_encoder:
        for n, value in sam.image_encoder.named_parameters():
            value.requires_grad = False
    elif args.finetune_type == 'lora':
        sam = LoRA_Sam(args, sam, r=4).sam

    sam.to('cuda')

    # === Dynamically get encoder output shape (C, H, W) using a dummy input ===
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 1024, 1024)).cuda()
        dummy_emb = sam.image_encoder(dummy)  # [1, C, H, W]
        C, H, W = dummy_emb.shape[1:]

    # === Define lightweight fusion MLPs ===
    # Map flattened SAM encoder features to CLIP dimension (512)
    fc_sam_to_clip = nn.Linear(C * H * W, 512).cuda()
    # Map fused CLIP-like feature back to channel space (C), then broadcast over H x W
    fc_fuse_to_decoder = nn.Linear(512, C).cuda()

    fusion_params = list(fc_sam_to_clip.parameters()) + list(fc_fuse_to_decoder.parameters())

    # === Optimizer and loss functions setup ===
    optimizer = optim.AdamW(list(sam.parameters()) + fusion_params,
                            lr=b_lr, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0.1, amsgrad=False)
    optimizer.zero_grad()
    # Scheduler is kept but step is handled by custom warmup / decay below
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion1 = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, to_onehot_y=True, reduction='mean')
    criterion2 = nn.CrossEntropyLoss()

    iter_num = 0
    max_iterations = epochs * len(trainloader)
    writer = SummaryWriter(dir_checkpoint + '/log')

    # === Training loop ===
    pbar = tqdm(range(epochs))
    val_largest_dsc = 0
    last_update_epoch = 0

    for epoch in pbar:
        sam.train()
        train_loss = 0.0

        for i, data in enumerate(tqdm(trainloader)):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data['mask']).cuda()

            # === Forward through SAM image encoder ===
            sam_img_emb = sam.image_encoder(imgs)  # [B, C, H, W]
            B, C_enc, H_enc, W_enc = sam_img_emb.shape
            assert C_enc == C and H_enc == H and W_enc == W, "Encoder output shape changed from dummy probing."

            # Global projection via flatten + Linear to CLIP dimension
            sam_feat = fc_sam_to_clip(sam_img_emb.view(B, -1))  # [B, 512]

            # === Encode images using frozen CLIP encoder ===
            imgs_pil = [transforms.ToPILImage()(img.cpu()) for img in imgs]
            clip_imgs = torch.stack([clip_preprocess(pil) for pil in imgs_pil]).to("cuda")
            with torch.no_grad():
                clip_feat = clip_model.encode_image(clip_imgs)  # [B, 512]
                clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

            # === Feature fusion in CLIP space (512 dim) ===
            text_feat_exp = clip_text_feat.expand(B, -1)  # [B, 512]
            # Simple additive fusion: SAM global feature + CLIP image feature + CLIP text feature
            merged_feat = sam_feat + clip_feat + text_feat_exp  # [B, 512]

            # === Project fused feature back to SAM encoder channel space and broadcast ===
            fuse_channel = fc_fuse_to_decoder(merged_feat)  # [B, C]
            fuse_map = fuse_channel.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]

            # Fuse by additive modulation on encoder feature map
            img_emb = sam_img_emb + fuse_map  # [B, C, H, W]

            # === Decode using SAM mask decoder ===
            sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=None, masks=None)
            pred, _ = sam.mask_decoder(
                image_embeddings=img_emb,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            # === Compute loss ===
            loss_dice = criterion1(pred, msks.float())
            loss_ce = criterion2(pred, torch.squeeze(msks.long(), 1))
            loss = loss_dice + loss_ce

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # === Learning rate scheduling (custom warmup + decay) ===
            if args.if_warmup and iter_num < args.warmup_period:
                lr_ = args.lr * ((iter_num + 1) / args.warmup_period)
            elif args.if_warmup:
                shift_iter = iter_num - args.warmup_period
                lr_ = args.lr * (1.0 - shift_iter / max_iterations) ** 0.9
            else:
                lr_ = args.lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            train_loss += loss.item()
            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)

        train_loss /= (i + 1)
        pbar.set_description(f"Epoch {epoch} | Train Loss: {train_loss:.4f}")

        # === Validation every 2 epochs ===
        if epoch % 2 == 0:
            eval_loss = 0.0
            dsc = 0.0
            sam.eval()

            with torch.no_grad():
                for i, data in enumerate(tqdm(valloader)):
                    imgs = data['image'].cuda()
                    msks = torchvision.transforms.Resize((args.out_size, args.out_size))(data['mask']).cuda()

                    sam_img_emb = sam.image_encoder(imgs)  # [B, C, H, W]
                    B, C_enc, H_enc, W_enc = sam_img_emb.shape
                    assert C_enc == C and H_enc == H and W_enc == W, "Encoder output shape changed from dummy probing."

                    sam_feat = fc_sam_to_clip(sam_img_emb.view(B, -1))  # [B, 512]

                    imgs_pil = [transforms.ToPILImage()(img.cpu()) for img in imgs]
                    clip_imgs = torch.stack([clip_preprocess(pil) for pil in imgs_pil]).to("cuda")
                    clip_feat = clip_model.encode_image(clip_imgs)  # [B, 512]
                    clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

                    text_feat_exp = clip_text_feat.expand(B, -1)  # [B, 512]
                    merged_feat = sam_feat + clip_feat + text_feat_exp  # [B, 512]

                    fuse_channel = fc_fuse_to_decoder(merged_feat)  # [B, C]
                    fuse_map = fuse_channel.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]

                    img_emb = sam_img_emb + fuse_map  # [B, C, H, W]

                    sparse_emb, dense_emb = sam.prompt_encoder(points=None, boxes=None, masks=None)
                    pred, _ = sam.mask_decoder(
                        image_embeddings=img_emb,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=True,
                    )

                    loss = criterion1(pred, msks.float()) + criterion2(pred, torch.squeeze(msks.long(), 1))
                    eval_loss += loss.item()

                    dsc_batch = dice_coeff_multi_class(
                        pred.argmax(1).cpu(),
                        torch.squeeze(msks.long(), 1).cpu(),
                        args.num_cls
                    )
                    dsc += dsc_batch

                eval_loss /= (i + 1)
                dsc /= (i + 1)
                writer.add_scalar('eval/loss', eval_loss, epoch)
                writer.add_scalar('eval/dice', dsc, epoch)

                print(f"Eval Epoch {epoch} | Val Loss: {eval_loss:.4f} | DSC: {dsc:.4f}")

                # === Save best checkpoint ===
                if dsc > val_largest_dsc:
                    val_largest_dsc = dsc
                    last_update_epoch = epoch
                    print(f"New best DSC: {dsc:.4f}")
                    torch.save({
                        'sam': sam.state_dict(),
                        'fc_sam_to_clip': fc_sam_to_clip.state_dict(),
                        'fc_fuse_to_decoder': fc_fuse_to_decoder.state_dict(),
                    }, dir_checkpoint + '/checkpoint_best.pth')
                elif epoch - last_update_epoch > 200:
                    print("Early stopping: no improvement in 200 epochs.")
                    break

    writer.close()


if __name__ == "__main__":
    dataset_name = args.dataset_name
    print(f"Training on dataset: {dataset_name}")
    print(f"Using text prompt for CLIP: \"{args.text_prompt}\"")

    train_img_list = args.train_img_list
    val_img_list = args.val_img_list

    Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.dir_checkpoint, "args.json"), 'w') as json_file:
        json.dump(vars(args), json_file, indent=4)

    train_dataset = Public_dataset(
        args, args.img_folder, args.mask_folder, train_img_list,
        phase='train', targets=[args.targets], normalize_type='sam', if_prompt=False
    )
    val_dataset = Public_dataset(
        args, args.img_folder, args.mask_folder, val_img_list,
        phase='val', targets=[args.targets], normalize_type='sam', if_prompt=False
    )

    trainloader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8)
    valloader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=8)

    train_model(trainloader, valloader, args.dir_checkpoint, args.epochs)
