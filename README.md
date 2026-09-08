# SAM-CLIP — Docker Release

[Yiyuan Lin](https://yiyuanlinxx.github.io/), Zachary Dashner, Ana Jimenez, Dustin Wilkerson, [Lance Cadle-Davidson](https://cals.cornell.edu/people/lance-cadle-davidson), [Summaira Riaz](https://vitisgen3.umn.edu/summaira-riaz), [Yu Jiang](https://cals.cornell.edu/people/yu-jiang)

[**Paper**](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6170008) | [**Dataset**](https://cornell.app.box.com/folder/359649298815?s=qkofzu5b24hqkev6y9raga9t9ihoc5l1) | [**Citation**](#citation) | [**Research branch**](https://github.com/YiyuanLinXX/SAM-CLIP/tree/main)

SAM-CLIP is the project accompanying *Integrating Large Multi-Modal Models for Automated Powdery Mildew Phenotyping in Grapevines*. The research integrates Segment Anything Model (SAM) with Contrastive Language-Image Pretraining (CLIP) embeddings for powdery mildew and canopy segmentation under field conditions. The imagery was collected with the autonomous phenotyping robot [PPBv2](https://github.com/YiyuanLinXX/PPBv2).

This `docker-release` branch packages the training, inference, and evaluation workflows described below. For the research overview and original workflow, see the `main` branch. The Docker commands accept image/mask pairs; they do not expose text prompts or a separate CLIP-embedding input.

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
git clone --branch docker-release https://github.com/YiyuanLinXX/SAM-CLIP.git
cd SAM-CLIP

# Create .env with your host UID/GID; preserve an existing configuration.
if [ ! -e .env ]; then
  printf 'HOST_UID=%s\nHOST_GID=%s\n' "$(id -u)" "$(id -g)" > .env
fi
docker compose build
docker compose run --rm sam-clip gpu-check
```

Run the remaining commands from the repository root. Host utilities used by these instructions include Git, `curl`, `sha256sum`, and Python 3 (for the synthetic dataset generator).

The GPU check prints Python, PyTorch, CUDA, cuDNN, and visible-GPU information, then performs a CUDA tensor operation.

Download an official SAM checkpoint:

```bash
./scripts/download_weights.sh vit_b
# or
./scripts/download_weights.sh vit_h
```

The download script verifies the checkpoint with SHA-256. These are initial SAM checkpoints for fine-tuning, not task-specific powdery mildew or canopy models. This branch does not bundle fine-tuned checkpoints; use a compatible trained checkpoint directory for inference.

If you customize `WEIGHTS_DIR`, pass that host directory explicitly to the download script; it does not read `.env`:

```bash
./scripts/download_weights.sh vit_b /absolute/path/to/weights
```

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

The defaults are defined in [compose.yaml](compose.yaml). The quick start creates a minimal `.env` with the current host UID/GID. Edit it to customize the mount paths using the example below, keeping your own UID/GID values. Create custom host directories before starting a container and ensure they are writable by that user.

The current checkout does not include `.env.example`, which `scripts/configure.sh` requires. Use the quick-start configuration command above for this release.

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

`HOST_UID` and `HOST_GID` make files written to mounted directories belong to the invoking host user instead of root. The Dockerfile copies only application code, dependency files, and container helpers into the image. This checkout does not include `.gitignore` or `.dockerignore`; keep runtime data out of commits and use external host directories for large datasets and checkpoints to avoid adding them to the Docker build context.

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
    ├── val.csv
    └── test.csv
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
  --path-root /workspace/data \
  --ratio 8 1 1 \
  --seed 42
```

The splitter pairs independently sorted image and mask paths, so both directories must contain equal numbers of corresponding files in the same sort order. Alternatively, write the CSV pairs explicitly if filenames do not align. Ensure the train and validation splits are nonempty.

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

### Training configuration

All training flags are defined in [app/cfg.py](app/cfg.py). Inspect them with `docker compose run --rm sam-clip train --help`. Boolean flags accept `true` or `false`; layer lists use comma-separated integers.

| Setting | Flag | Values / guidance |
|---|---|---|
| SAM backbone | `-arch` | `vit_b` or `vit_h` with matching downloaded weights; examples use ViT-B |
| Fine-tuning method | `-finetune_type` | `vanilla`, `adapter`, `lora` |
| Update encoder | `-if_update_encoder` | Enable when training encoder adapters or LoRA layers |
| Encoder / decoder adapters | `-if_encoder_adapter`, `-if_mask_decoder_adapter` | Used with adapter fine-tuning |
| Encoder adapter blocks | `-encoder_adapter_depths` | E.g. `0,1,10,11` for the ViT-B example |
| Encoder / decoder LoRA | `-if_encoder_lora_layer`, `-if_decoder_lora_layer` | Used with LoRA fine-tuning |
| Encoder LoRA blocks | `-encoder_lora_layer` | Comma-separated block indices |
| Learning rate | `-lr` | Default `0.001` |
| Warmup | `-if_warmup`, `-warmup_period` | Enable warmup and set its period |
| Batch / workers | `-b`, `-w` | Adjust for GPU memory and host resources |
| Labels | `-targets`, `-num_cls` | Binary or multiclass settings described above |

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

A checkpoint directory must contain both `args.json` and `checkpoint_best.pth` from the same training run. Mount it under `/workspace/checkpoints`. The inference examples expect a prepared folder of held-out images, `my_dataset/test_images`; inference reads a directory, not `test.csv`. Place matching ground-truth masks in `my_dataset/test_gt` for evaluation.

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

Semantic inference saves one PNG per input image using its filename stem and original image dimensions. Binary outputs use values 0 and 255; multiclass outputs retain class indices.

Inference uses the `normalize_type` saved in `args.json`. CPU inference is available with `--device cpu`, although it is much slower. The provided Compose services still reserve NVIDIA GPUs, even when the application is given `--device cpu`.

## Evaluation

Prediction and ground-truth files must share the same filename stem and label encoding. For binary semantic evaluation, use 0/255 ground-truth masks to match inference output; 0/1 ground truth must be converted first (see `convert-masks --help`).

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

See [VALIDATION.md](VALIDATION.md) for the recorded checks performed on this distribution. The recorded GPU training smoke test covers single-GPU ViT-B; it does not document a completed multi-GPU DDP training run.

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

DataLoader/shared-memory failure: reduce `-w` or check host shared-memory availability. Compose currently uses `ipc: host`; to use a private container shared-memory allocation controlled by `SHM_SIZE`, remove `ipc: host` from the shared service configuration first.

Missing images or masks: use comma-separated CSV files with no header, and paths relative to `/workspace/data`. Arguments passed to the application must use container paths, not host paths.

## License

This distribution is licensed under the Apache License 2.0 in `LICENSE`.

## Acknowledgements

Our framework builds on [finetuneSAM](https://github.com/mazurowski-lab/finetune-SAM). The research extends this foundation for multi-modal input, CLIP image and text embeddings, parameter-efficient fine-tuning, and high-throughput plant phenotyping.

Other foundations for this codebase include:

- [SAM](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [Medical SAM Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)
- [LoRA for SAM](https://github.com/JamesQFreeman/Sam_LoRA)

Follow the licenses and citation requirements of the relevant upstream projects when using or redistributing this software.

## Citation

Please cite our work if you use this code or the associated research:

```bib
@article{Lin2026,
  title = {Integrating Large Multi-Modal Models for Automated Powdery Mildew Phenotyping in Grapevines},
  url = {http://dx.doi.org/10.2139/ssrn.6170008},
  DOI = {10.2139/ssrn.6170008},
  publisher = {Elsevier BV},
  author = {Lin,  Yiyuan and Dashner,  Zachary and Jimenez,  Ana and Wilkerson,  Dustin and Cadle-Davidson,  Lance  E. and Riaz,  Summaira and Jiang,  Yu},
  year = {2026}
}

@inproceedings{linEffectiveIntegrationVision2024,
  title = {Effective Integration of Vision Foundational Models for Semantic Segmentation to Quantify Grape Foliage Powdery Mildew Infection},
  booktitle = {2024 ASABE Annual International Meeting},
  author = {Lin, Yiyuan and Underhill, Anna and Cadle-Davidson, Lance and Jimenez, Ana and Riaz, Summaira and Jiang, Yu},
  year = 2024,
  series = {ASABE Paper No. 2401108},
  pages = {1-12},
  publisher = {ASABE},
  address = {St. Joseph, MI},
  doi = {10.13031/aim.202401108},
  keywords = {CLIP,Computer Vision,Powdery Mildew,SAM,Semantic Segmentation,Vineyard Management.}
}
```
