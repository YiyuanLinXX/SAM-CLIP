import os
import argparse
import csv
import random
from pathlib import Path

def split_dataset(img_dir, mask_dir, output_dir, ratio=(8, 1, 1), seed=42, path_root=None):
    # Get image and mask file paths
    img_paths = sorted(list(Path(img_dir).glob("*")))
    mask_paths = sorted(list(Path(mask_dir).glob("*")))

    # Check if the number of images and masks are the same
    assert len(img_paths) == len(mask_paths), "Mismatch between image and mask counts!"

    # Shuffle the dataset
    combined = list(zip(img_paths, mask_paths))
    random.seed(seed)
    random.shuffle(combined)

    # Calculate split sizes
    total = len(combined)
    n_train = int(ratio[0] / sum(ratio) * total)
    n_val   = int(ratio[1] / sum(ratio) * total)
    n_test  = total - n_train - n_val

    train_set = combined[:n_train]
    val_set   = combined[n_train:n_train + n_val]
    test_set  = combined[n_train + n_val:]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Store paths relative to a stable mount root so CSV files remain portable
    # between machines. For /data/name/{images,masks}, the inferred root is
    # /data and rows become name/images/x.png,name/masks/x.png.
    if path_root is None:
        common_dataset_dir = Path(os.path.commonpath([Path(img_dir).resolve(), Path(mask_dir).resolve()]))
        path_root = common_dataset_dir.parent
    path_root = Path(path_root).resolve()

    # Write data splits to CSV files (comma-separated, no header)
    def write_csv(pairs, filename):
        with open(os.path.join(output_dir, filename), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for img_path, mask_path in pairs:
                try:
                    image_value = img_path.resolve().relative_to(path_root)
                    mask_value = mask_path.resolve().relative_to(path_root)
                except ValueError as error:
                    raise ValueError(
                        f"Both image and mask paths must be under path_root={path_root}"
                    ) from error
                writer.writerow([str(image_value), str(mask_value)])

    write_csv(train_set, 'train.csv')
    write_csv(val_set, 'val.csv')
    write_csv(test_set, 'test.csv')

    print(f"Split completed! Total samples: {total}")
    print(f"Train: {n_train}, Validation: {n_val}, Test: {n_test}")
    print(f"CSV files saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test CSV splits.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--path-root",
        default=None,
        help="Root used to make CSV paths relative (default: parent of common dataset directory)",
    )
    parser.add_argument("--ratio", type=int, nargs=3, default=(8, 1, 1), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    cli_args = parser.parse_args()
    split_dataset(
        img_dir=cli_args.image_dir,
        mask_dir=cli_args.mask_dir,
        output_dir=cli_args.output_dir,
        ratio=tuple(cli_args.ratio),
        seed=cli_args.seed,
        path_root=cli_args.path_root,
    )
