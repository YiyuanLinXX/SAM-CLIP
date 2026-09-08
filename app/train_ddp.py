#!/usr/bin/env python3
"""Standard one-process-per-GPU DDP training for SAM_CLIP."""

import json
import os
from pathlib import Path

import monai
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import cfg
from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
from utils.dataset import Public_dataset
from utils.dsc import dice_coeff_multi_class


class NoPromptModel(nn.Module):
    """Expose the no-prompt training path through DDP's forward call."""

    def __init__(self, sam):
        super().__init__()
        self.sam = sam

    def forward(self, images):
        image_embeddings = self.sam.image_encoder(images)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
        )
        return masks


def configure_finetuning(args, sam):
    if args.finetune_type == "adapter":
        for name, parameter in sam.named_parameters():
            if "Adapter" not in name:
                parameter.requires_grad = False
    elif args.finetune_type == "vanilla" and not args.if_update_encoder:
        for parameter in sam.image_encoder.parameters():
            parameter.requires_grad = False
    elif args.finetune_type == "lora":
        sam = LoRA_Sam(args, sam, r=4).sam
    return sam


def build_loaders(args, rank, world_size):
    train_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        args.train_img_list,
        phase="train",
        targets=[args.targets],
        normalize_type=args.normalize_type,
        if_prompt=False,
    )
    val_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        args.val_img_list,
        phase="val",
        targets=[args.targets],
        normalize_type=args.normalize_type,
        if_prompt=False,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("No training samples remained after loading/filtering the training CSV.")
    if len(val_dataset) == 0:
        raise RuntimeError("No validation samples remained after loading/filtering the validation CSV.")
    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=args.s
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        sampler=sampler,
        num_workers=args.w,
        pin_memory=True,
        persistent_workers=args.w > 0,
    )
    # Validation runs only on rank 0 to avoid duplicate samples and simplify
    # deterministic checkpoint selection.
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.b,
            shuffle=False,
            num_workers=args.w,
            pin_memory=True,
            persistent_workers=args.w > 0,
        )
    return train_loader, val_loader, sampler


@torch.no_grad()
def validate(model, loader, args, device, dice_loss, ce_loss):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    for data in tqdm(loader, desc="validation", leave=False):
        images = data["image"].to(device, non_blocking=True)
        masks = torchvision.transforms.Resize((args.out_size, args.out_size))(
            data["mask"]
        ).to(device, non_blocking=True)
        predictions = model(images)
        loss = dice_loss(predictions, masks.float()) + ce_loss(
            predictions, masks.squeeze(1).long()
        )
        score = dice_coeff_multi_class(
            predictions.argmax(dim=1).cpu(), masks.squeeze(1).cpu().long(), args.num_cls
        )
        total_loss += loss.item()
        total_dice += score.item()
        batches += 1
    return total_loss / batches, total_dice / batches


def main():
    args = cfg.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DDP training")
    if args.if_split_encoder_gpus:
        raise ValueError(
            "-if_split_encoder_gpus cannot be combined with train-ddp. "
            "DDP already assigns one complete model replica per GPU."
        )

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    checkpoint_dir = Path(args.dir_checkpoint)
    if rank == 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (checkpoint_dir / "args.json").open("w") as handle:
            json.dump(vars(args), handle, indent=2)
        print(f"DDP world size: {world_size}; checkpoint dir: {checkpoint_dir}")
    dist.barrier()

    train_loader, val_loader, sampler = build_loaders(args, rank, world_size)
    sam = sam_model_registry[args.arch](
        args, checkpoint=args.sam_ckpt, num_classes=args.num_cls
    )
    sam = configure_finetuning(args, sam).to(device)
    model = DistributedDataParallel(
        NoPromptModel(sam),
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    initial_lr = args.lr / args.warmup_period if args.if_warmup else args.lr
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(parameters, lr=initial_lr, weight_decay=0.1)
    dice_loss = monai.losses.DiceLoss(
        sigmoid=True, squared_pred=True, to_onehot_y=True, reduction="mean"
    )
    ce_loss = nn.CrossEntropyLoss()
    writer = SummaryWriter(str(checkpoint_dir / "log")) if rank == 0 else None
    best_dice = -1.0
    iteration = 0
    max_iterations = max(1, args.epochs * len(train_loader))

    try:
        for epoch in range(args.epochs):
            sampler.set_epoch(epoch)
            model.train()
            iterator = tqdm(
                train_loader,
                desc=f"epoch {epoch + 1}/{args.epochs}",
                disable=rank != 0,
            )
            for data in iterator:
                images = data["image"].to(device, non_blocking=True)
                masks = torchvision.transforms.Resize((args.out_size, args.out_size))(
                    data["mask"]
                ).to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                predictions = model(images)
                loss_dice = dice_loss(predictions, masks.float())
                loss_ce = ce_loss(predictions, masks.squeeze(1).long())
                loss = loss_dice + loss_ce
                loss.backward()
                optimizer.step()

                if args.if_warmup and iteration < args.warmup_period:
                    learning_rate = args.lr * ((iteration + 1) / args.warmup_period)
                elif args.if_warmup:
                    progress = (iteration - args.warmup_period) / max_iterations
                    learning_rate = args.lr * max(0.0, 1.0 - progress) ** 0.9
                else:
                    learning_rate = args.lr
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate

                if writer is not None:
                    writer.add_scalar("info/lr", learning_rate, iteration)
                    writer.add_scalar("info/total_loss", loss.item(), iteration)
                    writer.add_scalar("info/loss_ce", loss_ce.item(), iteration)
                    writer.add_scalar("info/loss_dice", loss_dice.item(), iteration)
                    iterator.set_postfix(loss=f"{loss.item():.4f}")
                iteration += 1

            dist.barrier()
            if rank == 0 and epoch % 2 == 0:
                val_loss, val_dice = validate(
                    model.module, val_loader, args, device, dice_loss, ce_loss
                )
                writer.add_scalar("eval/loss", val_loss, epoch)
                writer.add_scalar("eval/dice", val_dice, epoch)
                print(f"epoch={epoch} val_loss={val_loss:.6f} dice={val_dice:.6f}")
                if val_dice > best_dice:
                    best_dice = val_dice
                    torch.save(model.module.sam.state_dict(), checkpoint_dir / "checkpoint_best.pth")
                    print(f"saved new best checkpoint (dice={best_dice:.6f})")
            dist.barrier()
    finally:
        if writer is not None:
            writer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
