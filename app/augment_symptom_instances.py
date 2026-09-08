#!/usr/bin/env python3
"""Create canopy augmentation dataset with pasted PM/DM lesions."""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class LesionPatch:
    """Holds a cropped lesion instance along with metadata."""

    image: np.ndarray
    mask: np.ndarray  # bool mask
    label: str  # "PM" or "DM"
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paste PM/DM symptom instances onto canopy areas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--canopy-root",
        type=Path,
        default=Path("datasets/Canopy_Berry_2025_filtered"),
        help="Folder that contains canopy images/ and masks/ sub-directories.",
    )
    parser.add_argument(
        "--pm-root",
        type=Path,
        default=Path("datasets/PM_2019"),
        help="Powdery mildew dataset root with images/ and masks/ sub-directories.",
    )
    parser.add_argument(
        "--dm-root",
        type=Path,
        default=Path("datasets/DM_2021"),
        help="Downy mildew dataset root with images/ and masks/ sub-directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/Canopy_Berry_2025_augmented"),
        help="Destination directory where augmented images and masks will be saved.",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=1,
        help="Number of augmented copies to create per canopy image.",
    )
    parser.add_argument(
        "--min-paste",
        type=int,
        default=1,
        help="Minimum number of lesions to paste per augmented image.",
    )
    parser.add_argument(
        "--max-paste",
        type=int,
        default=20,
        help="Maximum number of lesions to paste per augmented image.",
    )
    parser.add_argument(
        "--scale-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.8, 1.2),
        help="Uniform scale range applied to lesion crops before pasting.",
    )
    parser.add_argument(
        "--instance-padding",
        type=int,
        default=4,
        help="Padding (in pixels) included around each cropped lesion instance.",
    )
    parser.add_argument(
        "--min-instance-area",
        type=int,
        default=100,
        help="Minimum number of foreground pixels for an extracted lesion instance.",
    )
    parser.add_argument(
        "--location-tries",
        type=int,
        default=40,
        help="Max attempts to find a valid canopy location for each lesion paste.",
    )
    parser.add_argument(
        "--canopy-class",
        type=int,
        default=1,
        help="Pixel value that denotes canopy regions in the canopy mask.",
    )
    parser.add_argument(
        "--dm-class",
        type=int,
        default=3,
        help="Pixel value assigned to pasted downy mildew regions in the output mask.",
    )
    parser.add_argument(
        "--pm-class",
        type=int,
        default=4,
        help="Pixel value assigned to pasted powdery mildew regions in the output mask.",
    )

    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=0,
        help="Mask pixels greater than this threshold are considered foreground.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Optional JSON file to store augmentation metadata (auto-set if omitted).",
    )
    parser.add_argument(
        "--density-alpha",
        type=float,
        default=0.5,
        help="Shape parameter (alpha) for Beta distribution used to randomize lesion density.",
    )
    parser.add_argument(
        "--density-beta",
        type=float,
        default=0.5,
        help="Shape parameter (beta) for Beta distribution used to randomize lesion density.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="How often (in augmented samples) to print progress. Set <=0 to disable.",
    )
    return parser.parse_args()


def list_pairs(root: Path) -> Iterable[Tuple[Path, Path]]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        return []
    for mask_path in sorted(mask_dir.glob("*.png")):
        image_path = image_dir / mask_path.name
        if not image_path.exists():
            continue
        yield image_path, mask_path


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), copy=False)


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def neighbors(y: int, x: int) -> Iterable[Tuple[int, int]]:
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            yield y + dy, x + dx


def connected_components(binary_mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = binary_mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not binary_mask[y, x] or visited[y, x]:
                continue
            q: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            coords: List[Tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                coords.append((cy, cx))
                for ny, nx in neighbors(cy, cx):
                    if 0 <= ny < h and 0 <= nx < w and binary_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            comps.append(coords)
    return comps


def extract_instances(
    dataset_root: Path,
    label_name: str,
    args: argparse.Namespace,
) -> List[LesionPatch]:
    patches: List[LesionPatch] = []
    for image_path, mask_path in list_pairs(dataset_root):
        mask_arr = load_mask(mask_path)
        binary = (mask_arr > args.mask_threshold).astype(np.uint8)
        if not np.any(binary):
            continue
        comps = connected_components(binary)
        if not comps:
            continue
        image_arr = load_image(image_path)
        h, w = image_arr.shape[:2]
        for coords in comps:
            if len(coords) < args.min_instance_area:
                continue
            ys, xs = zip(*coords)
            y0 = max(min(ys) - args.instance_padding, 0)
            x0 = max(min(xs) - args.instance_padding, 0)
            y1 = min(max(ys) + args.instance_padding + 1, h)
            x1 = min(max(xs) + args.instance_padding + 1, w)
            patch_img = image_arr[y0:y1, x0:x1]
            patch_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            for y, x in coords:
                patch_mask[y - y0, x - x0] = True
            patches.append(
                LesionPatch(
                    image=patch_img.copy(),
                    mask=patch_mask,
                    label=label_name,
                    source=image_path.name,
                )
            )
    return patches


def resize_patch(patch: LesionPatch, scale: float) -> Tuple[np.ndarray, np.ndarray]:
    if abs(scale - 1.0) < 1e-3:
        return patch.image, patch.mask
    image = Image.fromarray(patch.image)
    mask = Image.fromarray(patch.mask.astype(np.uint8) * 255)
    w = max(1, int(round(image.width * scale)))
    h = max(1, int(round(image.height * scale)))
    resized_img = np.array(image.resize((w, h), Image.BICUBIC))
    resized_mask = np.array(mask.resize((w, h), Image.NEAREST)) > 0
    return resized_img, resized_mask


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, mask_dir


def pick_patch(rng: random.Random, patches: Sequence[LesionPatch]) -> LesionPatch:
    return patches[rng.randrange(len(patches))]


def try_place_patch(
    rng: random.Random,
    base_image: np.ndarray,
    base_mask: np.ndarray,
    lesion_img: np.ndarray,
    lesion_mask: np.ndarray,
    lesion_value: int,
    canopy_class: int,
    max_tries: int,
) -> Tuple[bool, dict]:
    h, w = base_image.shape[:2]
    ph, pw = lesion_img.shape[:2]
    if ph >= h or pw >= w:
        return False, {}
    mask_pixels = np.argwhere(lesion_mask)
    if mask_pixels.size == 0:
        return False, {}
    for attempt in range(max_tries):
        y0 = rng.randint(0, h - ph)
        x0 = rng.randint(0, w - pw)
        region = base_mask[y0 : y0 + ph, x0 : x0 + pw]
        if np.all(region[lesion_mask] == canopy_class):
            region_img = base_image[y0 : y0 + ph, x0 : x0 + pw]
            over_mask = lesion_mask[..., None]
            np.copyto(region_img, lesion_img, where=over_mask)
            region_mask = base_mask[y0 : y0 + ph, x0 : x0 + pw]
            region_mask[lesion_mask] = lesion_value
            return True, {"y0": y0, "x0": x0, "height": ph, "width": pw}
    return False, {}


def sample_paste_count(rng: random.Random, args: argparse.Namespace) -> int:
    if args.min_paste == args.max_paste:
        return args.min_paste
    if args.density_alpha <= 0 or args.density_beta <= 0:
        raise ValueError("density-alpha and density-beta must be > 0")
    fraction = rng.betavariate(args.density_alpha, args.density_beta)
    span = args.max_paste - args.min_paste + 1
    count = args.min_paste + int(fraction * span)
    return min(args.max_paste, count)


def augment_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    pm_instances = extract_instances(args.pm_root, "PM", args)
    dm_instances = extract_instances(args.dm_root, "DM", args)
    lesion_bank = pm_instances + dm_instances
    if not lesion_bank:
        raise RuntimeError("No lesion instances found in PM or DM datasets.")

    print(f"Loaded {len(pm_instances)} PM and {len(dm_instances)} DM lesion instances.")
    image_out_dir, mask_out_dir = ensure_output_dirs(args.output_root)
    metadata_path = (
        args.metadata_path if args.metadata_path else args.output_root / "metadata.json"
    )
    placements = []

    canopy_pairs = list(list_pairs(args.canopy_root))
    if not canopy_pairs:
        raise RuntimeError(f"No canopy image/mask pairs found under {args.canopy_root}")

    total_jobs = len(canopy_pairs) * args.copies_per_image
    completed = 0

    for image_path, mask_path in canopy_pairs:
        base_image = load_image(image_path)
        canopy_mask = load_mask(mask_path)
        for copy_idx in range(args.copies_per_image):
            aug_image = base_image.copy()
            aug_mask = canopy_mask.copy()
            n_paste = sample_paste_count(rng, args)
            record = {
                "source_image": image_path.name,
                "source_mask": mask_path.name,
                "copy_index": copy_idx,
                "placements": [],
            }
            for _ in range(n_paste):
                lesion_patch = pick_patch(rng, lesion_bank)
                scale = rng.uniform(*args.scale_range)
                lesion_img, lesion_mask = resize_patch(lesion_patch, scale)
                lesion_value = args.pm_class if lesion_patch.label == "PM" else args.dm_class
                placed, loc = try_place_patch(
                    rng=rng,
                    base_image=aug_image,
                    base_mask=aug_mask,
                    lesion_img=lesion_img,
                    lesion_mask=lesion_mask,
                    lesion_value=lesion_value,
                    canopy_class=args.canopy_class,
                    max_tries=args.location_tries,
                )
                if placed:
                    record["placements"].append(
                        {
                            "label": lesion_patch.label,
                            "source": lesion_patch.source,
                            "scale": scale,
                            **loc,
                        }
                    )
            output_name = f"{image_path.stem}_aug{copy_idx:02d}.png"
            Image.fromarray(aug_image).save(image_out_dir / output_name)
            Image.fromarray(aug_mask).save(mask_out_dir / output_name)
            record["output_file"] = output_name
            placements.append(record)
            completed += 1
            if args.log_interval > 0 and completed % args.log_interval == 0:
                print(f"[{completed}/{total_jobs}] Wrote {output_name}", flush=True)

    with open(metadata_path, "w", encoding="utf-8") as fout:
        json.dump(placements, fout, indent=2)
    print(f"Saved augmented dataset to {args.output_root} and metadata to {metadata_path}.")


def main() -> None:
    args = parse_args()
    if args.min_paste > args.max_paste:
        raise ValueError("min-paste must be <= max-paste")
    if args.scale_range[0] <= 0 or args.scale_range[1] <= 0:
        raise ValueError("scale-range values must be positive")
    augment_dataset(args)


if __name__ == "__main__":
    main()
