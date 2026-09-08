#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 sam-clip train \
  -arch vit_b \
  -finetune_type adapter \
  -if_update_encoder true \
  -if_encoder_adapter true \
  -if_mask_decoder_adapter true \
  -encoder_adapter_depths 0,1,10,11 \
  -sam_ckpt /workspace/weights/sam_vit_b_01ec64.pth \
  -img_folder /workspace/data \
  -mask_folder /workspace/data \
  -train_img_list /workspace/data/my_dataset/train.csv \
  -val_img_list /workspace/data/my_dataset/val.csv \
  -dataset_name my_dataset \
  -targets combine_all \
  -num_cls 2 \
  -dir_checkpoint /workspace/checkpoints/my_experiment \
  -epochs 200 \
  -b 2 \
  -w 4 \
  -if_warmup true
