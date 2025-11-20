# SAM-CLIP
Authors: Yiyuan Lin, Zachary Dashner, Ana Jimenez, Dustin Wilkerson, Lance Cadle-Davidson, Summaira Riaz, Yu Jiang

[[`Paper`]()] [[`Dataset`]()] [[`BibTex`](#Citation)]

This is the official code for our paper: <TODO>,



This is the official implementation of **SAM-CLIP**, a large multi-modal model where Segment Anything Model extended with CLIP embeddings for improved segmentation performance.

<p float="left">
  <img src="assets/model_arch.png" width="80%" />
</p>



## Pre-requisites

**Environment Setup**

We recommend using Conda:

```bash
conda env create -f environment.yml #if needed, edit the .yml file (e.g. cuda version) to adapt it to your hardware
conda activate sam_clip
```

You can also use pip:

```bash
pip install -r requirements.txt
```



## Instruction

### Step 1: Dataset and Model Weights Preparation

- Prepare your image-mask pairs under the following structure:

  ```kotlin
  datasets/
  └── <your_dataset_name>/
      ├── images/
      │   ├── sample_001.png
      │   ├── ...
      ├── masks/
      │   ├── sample_001_mask.png
      │   ├── ...
      ├── train.csv
      ├── val.csv
      └── test.csv
  
  ```

  You can use the `datasets/splitDatasets.py` to prepare the `.csv` files. Each `.csv` should have the following format:

  ```csv
  images/sample_001.png,masks/sample_001_mask.png
  images/sample_002.png,masks/sample_002_mask.png
  ...
  ```

- Download the pre-trained weights of SAM from the [official SAM repo](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) or our fine-tuned checkpoints for [PM]() and [Canopy](), and put them under `weights`.





### Step 2: Training







### Step 3: Inference & Evaluation

1. Run inference with:

   ```bash
   bash inference_sam_clip.bash # edit file path in the .sh file before running
   ```

   The masks will be saved in the `output_dir` you set in  `inference_sam_clip.bash`.

2. Run comprehensive performance evaluation ($mIoU^I$, $mIoU^D$, $mIoU^{Cq}$) with:

   ```bash
   bash eval_all.bash # edit FOLDER_PAIRS, QUANTILES, etc before running
   ```

   The evaluation results will







## Acknowledgement

Our framework is developed on top of the [finetuneSAM](https://github.com/mazurowski-lab/finetune-SAM) codebase, with major modifications to support:

- Multi-modal input
- CLIP-based image and/or text embeddings
- Unified backbone with parameter-efficient finetuning
- Specialized datasets in high-throughput plant phenotyping

Other foundations for this codebase:

1. [SAM](https://github.com/facebookresearch/segment-anything)
2. [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
3. [MedSAM](https://github.com/bowang-lab/MedSAM)
4. [Medical SAM Adapter](https://github.com/KidsWithTokens/Medical-SAM-Adapter)
5. [LoRA for SAM](https://github.com/JamesQFreeman/Sam_LoRA)



## Citation

Please cite our paper if you find our codes or paper helpful

```bib
TODO
```

