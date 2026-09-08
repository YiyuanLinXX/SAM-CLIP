# SAM_CLIP Docker

English | [简体中文](README.zh-CN.md)

A reproducible NVIDIA GPU container for SAM_CLIP training, multi-GPU DDP training, semantic-segmentation inference, instance-scored inference, evaluation, TensorBoard, and dataset utilities.

The image contains application code and pinned dependencies only. Datasets, pretrained SAM weights, checkpoints, and outputs are mounted at runtime and are intentionally excluded from the image and repository.

## Features

- NVIDIA GPU support through the NVIDIA Container Toolkit
- Reproducible PyTorch 1.13.1 + CUDA 11.6 + cuDNN 8 environment
- Single-GPU training
- Standard one-process-per-GPU `torchrun` DDP training
- Semantic-mask inference
- Probability-map and connected-component instance inference
- Semantic mIoU, worst-case, and instance AP evaluation
- TensorBoard
- Dataset splitting, mask conversion, COCO mask export, augmentation, and visualization tools
- A self-contained synthetic-data smoke test

## Requirements

- Linux x86_64 host
- NVIDIA GPU
- NVIDIA driver 510.39.01 or newer for CUDA 11.6
- Docker Engine with Docker Compose v2
- NVIDIA Container Toolkit configured for Docker

Installation references:

- [Docker Engine installation](https://docs.docker.com/engine/install/)
- [Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Verify the host before building:

```bash
nvidia-smi
docker version
docker compose version
docker run --rm --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Quick start

```bash
git clone <repository-url> SAM_CLIP_docker
cd SAM_CLIP_docker
./scripts/configure.sh
docker compose build
docker compose run --rm sam-clip gpu-check
```

The GPU check prints Python, PyTorch, CUDA, cuDNN, and visible-GPU information, then performs a CUDA tensor operation.

Download an official SAM checkpoint:

```bash
./scripts/download_weights.sh vit_b
# or
./scripts/download_weights.sh vit_h
```

The download script verifies the checkpoint with SHA-256.

Run the complete single-GPU smoke test with a generated synthetic dataset:

```bash
./examples/smoke-test-single-gpu.sh
```

It uses GPU 0 by default and trains ViT-B for one epoch on two generated training pairs and two validation pairs. Select another GPU and output name with:

```bash
GPU_ID=1 SMOKE_RUN_NAME=my_smoke_test \
  ./examples/smoke-test-single-gpu.sh
```

## Runtime directories

Docker Compose mounts four directories:

| Container path | Purpose | Default host path |
|---|---|---|
| `/workspace/data` | Datasets | `./volumes/data` |
| `/workspace/weights` | Initial SAM weights | `./volumes/weights` |
| `/workspace/checkpoints` | Training checkpoints and logs | `./volumes/checkpoints` |
| `/workspace/outputs` | Inference and evaluation output | `./volumes/outputs` |

The defaults are defined in `.env.example`. Generate `.env` with the current host UID/GID, then edit paths when needed:

```bash
./scripts/configure.sh
```

Example:

```dotenv
SAM_CLIP_IMAGE=sam-clip:cuda11.6
HOST_UID=1000
HOST_GID=1000
DATA_DIR=/absolute/path/to/data
WEIGHTS_DIR=/absolute/path/to/weights
CHECKPOINTS_DIR=/absolute/path/to/checkpoints
OUTPUTS_DIR=/absolute/path/to/outputs
SHM_SIZE=16gb
TENSORBOARD_PORT=6006
```

`HOST_UID` and `HOST_GID` make files written to mounted directories belong to the invoking host user instead of root. `.env`, datasets, weights, checkpoints, and outputs are ignored by Git and the Docker build context.

## Dataset format

Prepare paired 2D images and single-channel masks:

```text
data/
└── my_dataset/
    ├── images/
    │   ├── sample_001.png
    │   └── sample_002.png
    ├── masks/
    │   ├── sample_001.png
    │   └── sample_002.png
    ├── train.csv
    └── val.csv
```

Each CSV is comma-separated with no header. Paths are relative to `/workspace/data`:

```csv
my_dataset/images/sample_001.png,my_dataset/masks/sample_001.png
my_dataset/images/sample_002.png,my_dataset/masks/sample_002.png
```

Create deterministic train/validation/test splits:

```bash
docker compose run --rm sam-clip split-dataset \
  --image-dir /workspace/data/my_dataset/images \
  --mask-dir /workspace/data/my_dataset/masks \
  --output-dir /workspace/data/my_dataset \
  --ratio 8 1 1 \
  --seed 42
```

Important label options:

- `-targets combine_all` converts all nonzero mask values into one foreground class. Use `-num_cls 2` in the usual binary case.
- `-targets multi_all` keeps class IDs. Use background ID 0, contiguous class IDs, and set `-num_cls` to the total number of classes including background.
- `-normalize_type sam` applies ImageNet/SAM normalization.
- `-normalize_type medsam` applies per-image `[0,1]` normalization and should only be used with compatible pretrained weights.

## Single-GPU training

Edit `examples/train-single-gpu.sh`, then run:

```bash
./examples/train-single-gpu.sh
```

Equivalent command:

```bash
docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES=0 \
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
```

The output directory contains:

```text
my_experiment/
├── args.json
├── checkpoint_best.pth
└── log/
```

ViT-H uses substantially more GPU memory than ViT-B. Reduce `-b` first if training runs out of memory.

## Multi-GPU DDP training

The DDP entrypoint launches one complete model replica per visible GPU. The `-b` value is the per-GPU batch size, so the global batch size is `-b × number of GPUs`.

Edit and run:

```bash
./examples/train-multi-gpu.sh
```

To choose specific GPUs:

```bash
docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  sam-clip train-ddp [training arguments]
```

The entrypoint derives `torchrun --nproc_per_node` from the number of visible GPUs. Override it with `-e NPROC_PER_NODE=2` only when that value does not exceed the visible GPU count.

## Inference

A checkpoint directory must contain both `args.json` and `checkpoint_best.pth`.

Semantic-mask inference:

```bash
docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES=0 \
  sam-clip infer \
  --checkpoint_dir /workspace/checkpoints/my_experiment \
  --image_dir /workspace/data/my_dataset/test_images \
  --output_dir /workspace/outputs/my_experiment
```

Inference with probability maps, connected-component instance masks, and confidence scores:

```bash
docker compose run --rm \
  -e CUDA_VISIBLE_DEVICES=0 \
  sam-clip infer-instances \
  --checkpoint_dir /workspace/checkpoints/my_experiment \
  --image_dir /workspace/data/my_dataset/test_images \
  --output_dir /workspace/outputs/my_experiment_instances
```

Inference uses the `normalize_type` saved in `args.json`. CPU inference is available with `--device cpu`, although it is much slower.

## Evaluation

Semantic mIoU, per-image metrics, and worst-quantile analysis:

```bash
docker compose run --rm sam-clip evaluate \
  --pred_folder /workspace/outputs/my_experiment \
  --gt_folder /workspace/data/my_dataset/test_gt \
  --output_csv /workspace/outputs/my_experiment/evaluation.csv \
  --save_per_image \
  --quantiles 5 10 15
```

Instance AP plus semantic evaluation, using output from `infer-instances`:

```bash
docker compose run --rm sam-clip evaluate-instances \
  --pred_dir /workspace/outputs/my_experiment_instances \
  --gt_dir /workspace/data/my_dataset/test_gt
```

## TensorBoard

```bash
docker compose up tensorboard
```

Open `http://localhost:6006`. Change `TENSORBOARD_PORT` in `.env` if required.

## Utility commands

List all container commands:

```bash
docker compose run --rm sam-clip help
```

Available utilities include:

```bash
docker compose run --rm sam-clip augment --help
docker compose run --rm sam-clip convert-masks --help
docker compose run --rm sam-clip export-coco --help
docker compose run --rm sam-clip extract-subset --help
docker compose run --rm sam-clip split-folder --help
docker compose run --rm sam-clip split-dataset --help
docker compose run --rm sam-clip colorize --help
```

Run an arbitrary packaged Python module or open a shell:

```bash
docker compose run --rm sam-clip python /opt/sam_clip/utils/eval_all.py --help
docker compose run --rm sam-clip shell
```

## Reproducibility

- Base image: `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime`
- The linux/amd64 base-image digest is pinned in `Dockerfile`
- PyTorch: 1.13.1
- torchvision: 0.14.1
- CUDA: 11.6
- MONAI: 1.2.0
- Direct Python dependencies: `requirements.txt`
- Transitive and framework compatibility constraints: `constraints.txt`

Record the built image ID when publishing a release:

```bash
docker image inspect sam-clip:cuda11.6 --format '{{.Id}}'
```

See `VALIDATION.md` for the checks performed on this distribution.

## Publishing the image

Push to a registry:

```bash
docker tag sam-clip:cuda11.6 registry.example.com/team/sam-clip:cuda11.6-v1
docker push registry.example.com/team/sam-clip:cuda11.6-v1
```

Offline transfer:

```bash
docker save sam-clip:cuda11.6 | gzip > sam-clip_cuda11.6-v1.tar.gz
gzip -dc sam-clip_cuda11.6-v1.tar.gz | docker load
```

Datasets, weights, checkpoints, and outputs must be distributed separately and mounted through `.env`.

## Troubleshooting

`could not select device driver "nvidia"`: install and configure the NVIDIA Container Toolkit, then restart Docker.

`CUDA driver version is insufficient`: upgrade the host NVIDIA driver or verify that the CUDA 11.6 image is being used.

No visible GPU: run `gpu-check`, then inspect `CUDA_VISIBLE_DEVICES` and the Docker NVIDIA runtime configuration.

DataLoader/shared-memory failure: increase `SHM_SIZE` in `.env` or reduce `-w`.

Missing images or masks: use comma-separated CSV files with no header, and paths relative to `/workspace/data`. Arguments passed to the application must use container paths, not host paths.

## License and acknowledgements

This distribution is licensed under the Apache License 2.0 in `LICENSE`.

The project contains or depends on work from Segment Anything, MobileSAM, MedSAM, Medical SAM Adapter, and LoRA for SAM. Follow the licenses and citation requirements of the relevant upstream projects when using or redistributing this software.
