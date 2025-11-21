export CUDA_VISIBLE_DEVICES="0"

python3 inference_sam_clip.py \
  --checkpoint_dir '/home/yl3663/SAM-CLIP/ckpt/PM_2019/SAM_CLIP_vit_h_ed_adapter_PM_2019' \
  --image_dir '/home/yl3663/SAM-CLIP/datasets/PM_2019/test_images' \
  --output_dir '/home/yl3663/SAM-CLIP/inference_results/SAM_CLIP_vit_h_ed_adapter_PM_2019' \
  --text_prompt "powdery mildew"
