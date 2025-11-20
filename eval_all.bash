#!/bin/bash
# ====================================
# Batch run mIoU evaluation + worst case metrics
# ====================================
#
# USAGE:
#   1) Edit FOLDER_PAIRS: each line "PRED|GT"
#   2) Edit the arguments "SAVE_PER_IMAGE", "QUANTILES" and "SCRIPT_PATH"
#   3) run the following command in the terminal
#
#            bash eval_all.bash
#
#
# ====================================

FOLDER_PAIRS=(
  # "PRED|GT"
    "/media/yl3663/Data/Datasets/PM_2019/test_image|/media/yl3663/Data/Datasets/PM_2019/test_gt"
    "/media/yl3663/Data/Datasets/Canopy_2020/test_image|/media/yl3663/Data/Datasets/Canopy_2020/test_gt"
)

SAVE_PER_IMAGE="--save_per_image"
QUANTILES="--quantiles 1 5 6 7 8 9 10 15 20"
SCRIPT_PATH="utils/eval_all.py"

clean_pair() {
  # remove CR, remove NBSP, trim leading/trailing spaces
  echo -n "$1" \
  | tr -d '\r' \
  | sed -e 's/\xC2\xA0//g' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

for raw in "${FOLDER_PAIRS[@]}"; do
  pair=$(clean_pair "$raw")

  # skip malformed lines
  if [[ "$pair" != *"|"* ]] || [[ -z "$pair" ]]; then
    echo ">> Skipping malformed pair: [$raw]"
    continue
  fi

  PRED_FOLDER="${pair%%|*}"
  GT_FOLDER="${pair##*|}"

  # trim again after split (safety)
  PRED_FOLDER=$(clean_pair "$PRED_FOLDER")
  GT_FOLDER=$(clean_pair "$GT_FOLDER")

  # sanity check
  if [[ ! -d "$PRED_FOLDER" ]]; then
    echo "!! Pred folder not found: $PRED_FOLDER — skipped."
    continue
  fi
  if [[ ! -d "$GT_FOLDER" ]]; then
    echo "!! GT folder not found: $GT_FOLDER — skipped."
    continue
  fi

  OUTPUT_CSV="${PRED_FOLDER}/evaluation_results.csv"

  echo "======================================="
  echo "Evaluating:"
  echo "Pred folder: $PRED_FOLDER"
  echo "GT folder:   $GT_FOLDER"
  echo "Output CSV:  $OUTPUT_CSV"
  echo "======================================="

  python3 "$SCRIPT_PATH" \
    --pred_folder "$PRED_FOLDER" \
    --gt_folder "$GT_FOLDER" \
    --output_csv "$OUTPUT_CSV" \
    $SAVE_PER_IMAGE \
    $QUANTILES
done

