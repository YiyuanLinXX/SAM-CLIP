#!/usr/bin/env bash
set -euo pipefail

docker compose run --rm -e CUDA_VISIBLE_DEVICES=0 sam-clip infer \
  --checkpoint_dir /workspace/checkpoints/my_experiment \
  --image_dir /workspace/data/my_dataset/test_images \
  --output_dir /workspace/outputs/my_experiment
