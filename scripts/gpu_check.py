#!/usr/bin/env python3
"""Fail-fast validation for the CUDA runtime exposed to this container."""

import platform
import sys

import torch


def main() -> int:
    print(f"Python:      {platform.python_version()}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Built CUDA:  {torch.version.cuda}")
    print(f"cuDNN:       {torch.backends.cudnn.version()}")
    print(f"CUDA ready:  {torch.cuda.is_available()}")

    count = torch.cuda.device_count()
    print(f"GPU count:   {count}")
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / 1024**3
        print(f"GPU {index}:      {properties.name} ({memory_gib:.1f} GiB)")

    if not torch.cuda.is_available() or count == 0:
        print(
            "ERROR: No NVIDIA GPU is available. Install the NVIDIA driver and "
            "NVIDIA Container Toolkit, then launch Docker with --gpus all.",
            file=sys.stderr,
        )
        return 1

    tensor = torch.tensor([1.0], device="cuda")
    print(f"CUDA test:   {(tensor + 1).item():.1f} (expected 2.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
