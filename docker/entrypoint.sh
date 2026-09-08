#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${SAM_CLIP_HOME:-/opt/sam_clip}"
mkdir -p "${HOME:-/tmp/sam-clip-home}"

show_help() {
  cat <<'EOF'
SAM_CLIP container commands

  gpu-check              Verify PyTorch, CUDA, cuDNN, and visible GPUs
  train [ARGS...]        Single-GPU training
  train-ddp [ARGS...]    Multi-GPU DDP training (launch with all requested GPUs)
  infer [ARGS...]        Semantic-mask inference
  infer-instances [...]  Inference with probability maps and instance scores
  evaluate [ARGS...]     Semantic mIoU and worst-case evaluation
  evaluate-instances ... Instance AP plus semantic evaluation
  augment [ARGS...]      Instance-copy data augmentation
  convert-masks [...]    Convert mask values 0/1 to 0/255
  export-coco [ARGS...]  Export semantic masks from COCO annotations
  extract-subset [...]   Extract a dataset subset
  split-folder [ARGS...] Copy image/mask pairs listed in a CSV
  split-dataset [...]    Create deterministic train/val/test CSV files
  colorize [ARGS...]     Colorize a mask (legacy maskVis utility)
  tensorboard [ARGS...]  Start TensorBoard
  python [ARGS...]       Run Python in the packaged source environment
  shell                  Start an interactive Bash shell

All paths passed to commands are container paths. Standard mounts are:
  /workspace/data, /workspace/weights, /workspace/checkpoints, /workspace/outputs

Use "sam-clip <command> --help" for command-specific options.
EOF
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command_name}" in
  help|-h|--help)
    show_help
    ;;
  gpu-check)
    exec python "${APP_DIR}/tools/gpu_check.py" "$@"
    ;;
  train)
    exec python "${APP_DIR}/SingleGPU_train_finetune_noprompt.py" "$@"
    ;;
  train-ddp)
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
      exec python "${APP_DIR}/train_ddp.py" "$@"
    fi
    if [[ -n "${NPROC_PER_NODE:-}" ]]; then
      nproc="${NPROC_PER_NODE}"
    else
      nproc="$(python -c 'import torch; print(torch.cuda.device_count())')"
    fi
    if [[ "${nproc}" -lt 2 ]]; then
      echo "train-ddp requires at least two visible GPUs; found ${nproc}." >&2
      exit 2
    fi
    exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${nproc}" "${APP_DIR}/train_ddp.py" "$@"
    ;;
  infer)
    exec python "${APP_DIR}/inference_sam_clip.py" "$@"
    ;;
  infer-instances)
    exec python "${APP_DIR}/inference_sam_clip_instance_score.py" "$@"
    ;;
  evaluate)
    exec python "${APP_DIR}/utils/eval_all.py" "$@"
    ;;
  evaluate-instances)
    exec python "${APP_DIR}/utils/eval_instance_ap.py" "$@"
    ;;
  augment)
    exec python "${APP_DIR}/augment_symptom_instances.py" "$@"
    ;;
  convert-masks)
    exec python "${APP_DIR}/utils/convert_masks.py" "$@"
    ;;
  export-coco)
    exec python "${APP_DIR}/utils/export_ldd_masks_from_coco.py" "$@"
    ;;
  extract-subset)
    exec python "${APP_DIR}/utils/extract_vl_powdery_mildew_subset.py" "$@"
    ;;
  split-folder)
    exec python "${APP_DIR}/dataset_tools/splitFolder.py" "$@"
    ;;
  split-dataset)
    exec python "${APP_DIR}/dataset_tools/splitDataset.py" "$@"
    ;;
  colorize)
    exec python "${APP_DIR}/maskVis.py" "$@"
    ;;
  tensorboard)
    exec tensorboard "$@"
    ;;
  python)
    exec python "$@"
    ;;
  shell|bash)
    exec /bin/bash "$@"
    ;;
  *)
    exec "${command_name}" "$@"
    ;;
esac
