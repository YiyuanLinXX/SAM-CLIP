#!/bin/bash
# ====================================
# Batch run mIoU evaluation + worst case metrics
# ====================================
#
#   1) Edit FOLDER_PAIRS: each line "PRED|GT"
#           FOLDER_PAIRS format:
#              FOLDER_PAIRS=(
#                  PRED="/pred_path_1" GT="/gt_path_1"
#                  PRED="/pred_path_2" GT="/gt_path_2"
#              )
#   2) Edit the arguments "SAVE_PER_IMAGE", "QUANTILES", "TARGET_LABELS" and "SCRIPT_PATH"
#   3) run the following command in the terminal
#
#            bash eval_all.bash
# ====================================

FOLDER_PAIRS=(
    PRED="/home/yl3663/SAM_CLIP/test_results/PM_2019/SAM_CLIP_vit_b_ed_adapter_PM_2019" GT="/home/yl3663/SAM_CLIP/datasets/PM_2019/test_gt"
)

# 1 = enable --save_per_image, 0 = disable
SAVE_PER_IMAGE=1

# Only values, not flag
QUANTILES="5 10 15"

# Optional: target label values for worst-case metrics.
# Leave empty ("") to auto-select all non-background (non-zero) classes.
# Otherwise specify raw pixel values, e.g.:
#   TARGET_LABELS="1"
#   TARGET_LABELS="1 3 255"
TARGET_LABELS=""

SCRIPT_PATH="./utils/eval_all.py"

# -----------------------------------------
# Main loop
# -----------------------------------------

index=0
total=${#FOLDER_PAIRS[@]}

while [[ $index -lt $total ]]; do

    # Parse two consecutive entries: PRED="xxx", GT="xxx"
    eval ${FOLDER_PAIRS[$index]}     # defines PRED
    eval ${FOLDER_PAIRS[$((index+1))]}  # defines GT

    PRED_FOLDER="$PRED"
    GT_FOLDER="$GT"

    # Increment index by 2
    index=$((index+2))

    # sanity check
    if [[ ! -d "$PRED_FOLDER" ]]; then
        echo "!! Pred folder not found: $PRED_FOLDER"
        continue
    fi
    if [[ ! -d "$GT_FOLDER" ]]; then
        echo "!! GT folder not found: $GT_FOLDER"
        continue
    fi

    OUTPUT_CSV="${PRED_FOLDER}/evaluation_results.csv"

    echo "======================================="
    echo "Evaluating:"
    echo "Pred folder: $PRED_FOLDER"
    echo "GT folder:   $GT_FOLDER"
    echo "Output CSV:  $OUTPUT_CSV"
    echo "======================================="

    # Build command (your preferred format)
    CMD=(python3 "$SCRIPT_PATH"
         --pred_folder "$PRED_FOLDER"
         --gt_folder "$GT_FOLDER"
         --output_csv "$OUTPUT_CSV"
    )

    # --save_per_image is a boolean flag
    if [[ "$SAVE_PER_IMAGE" -eq 1 ]]; then
        CMD+=(--save_per_image)
    fi

    # add quantiles
    if [[ -n "$QUANTILES" ]]; then
        CMD+=(--quantiles $QUANTILES)
    fi

    # add target labels
    if [[ -n "$TARGET_LABELS" ]]; then
        CMD+=(--target_labels $TARGET_LABELS)
    fi

    echo "Running:"
    printf "  %q " "${CMD[@]}"
    echo -e "\n"

    "${CMD[@]}"

done

