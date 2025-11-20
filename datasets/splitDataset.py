import os
import csv
import random
from pathlib import Path

def split_dataset(img_dir, mask_dir, output_dir, ratio=(6, 2, 2), seed=42):
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

    # Write data splits to CSV files (comma-separated)
    def write_csv(pairs, filename):
        with open(os.path.join(output_dir, filename), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for img_path, mask_path in pairs:
                writer.writerow([str(img_path), str(mask_path)])

    write_csv(train_set, 'train.csv')
    write_csv(val_set, 'val.csv')
    write_csv(test_set, 'test.csv')

    print(f"Split completed! Total samples: {total}")
    print(f"Train: {n_train}, Validation: {n_val}, Test: {n_test}")
    print(f"CSV files saved to: {output_dir}")

if __name__ == "__main__":
    split_dataset(
        img_dir="/home/yl3663/SAM-SLIP/datasets/PM_2019/images",
        mask_dir="/home/yl3663/SAM-CLIP/datasets/PM_2019/masks",
        output_dir="/home/yl3663/SAM-CLIP/datasets/PM_2019/",
        ratio=(6, 2, 2)
    )