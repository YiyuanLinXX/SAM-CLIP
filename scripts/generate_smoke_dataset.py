#!/usr/bin/env python3
"""Generate a tiny dependency-free image/mask dataset for the Docker smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path


WIDTH = 64
HEIGHT = 64


def write_ppm(path: Path, sample_index: int) -> None:
    pixels = bytearray()
    center_x = 18 + sample_index * 7
    center_y = 22 + sample_index * 5
    for y in range(HEIGHT):
        for x in range(WIDTH):
            foreground = (x - center_x) ** 2 + (y - center_y) ** 2 <= 11**2
            if foreground:
                pixels.extend((210, 55 + sample_index * 10, 45))
            else:
                pixels.extend((35, 95 + (x + y) % 30, 45))
    path.write_bytes(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode() + pixels)


def write_pgm(path: Path, sample_index: int) -> None:
    pixels = bytearray()
    center_x = 18 + sample_index * 7
    center_y = 22 + sample_index * 5
    for y in range(HEIGHT):
        for x in range(WIDTH):
            foreground = (x - center_x) ** 2 + (y - center_y) ** 2 <= 11**2
            pixels.append(255 if foreground else 0)
    path.write_bytes(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode() + pixels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(4):
        image_name = f"sample_{index}.ppm"
        mask_name = f"sample_{index}.pgm"
        write_ppm(image_dir / image_name, index)
        write_pgm(mask_dir / mask_name, index)
        rows.append(f"images/{image_name},masks/{mask_name}\n")

    (args.output_dir / "train.csv").write_text("".join(rows[:2]), encoding="utf-8")
    (args.output_dir / "val.csv").write_text("".join(rows[2:]), encoding="utf-8")
    print(f"Generated synthetic smoke-test data in {args.output_dir}")


if __name__ == "__main__":
    main()
