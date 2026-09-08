#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
gpu_id="${GPU_ID:-0}"
run_name="${SMOKE_RUN_NAME:-smoke_single_gpu}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/sam-clip-smoke.XXXXXX")"

cleanup() {
  rm -rf -- "${smoke_root}"
}
trap cleanup EXIT

cd "${project_dir}"

python3 "${project_dir}/scripts/generate_smoke_dataset.py" "${smoke_root}/data"

if ! docker compose run --rm sam-clip test -f /workspace/weights/sam_vit_b_01ec64.pth; then
  echo "Missing ViT-B weights. Run: ./scripts/download_weights.sh vit_b" >&2
  exit 2
fi

docker compose run --rm -e CUDA_VISIBLE_DEVICES="${gpu_id}" sam-clip gpu-check

DATA_DIR="${smoke_root}/data" docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES="${gpu_id}" \
  sam-clip train \
  -arch vit_b \
  -finetune_type adapter \
  -if_update_encoder true \
  -if_encoder_adapter true \
  -if_mask_decoder_adapter true \
  -encoder_adapter_depths 0,1,10,11 \
  -sam_ckpt /workspace/weights/sam_vit_b_01ec64.pth \
  -img_folder /workspace/data \
  -mask_folder /workspace/data \
  -train_img_list /workspace/data/train.csv \
  -val_img_list /workspace/data/val.csv \
  -dataset_name synthetic_smoke \
  -targets combine_all \
  -normalize_type sam \
  -num_cls 2 \
  -dir_checkpoint "/workspace/checkpoints/${run_name}" \
  -epochs 1 \
  -b 1 \
  -w 0 \
  -if_warmup false

echo "Smoke test passed. Artifacts: /workspace/checkpoints/${run_name}"
