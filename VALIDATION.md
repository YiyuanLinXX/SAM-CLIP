# Validation status

Completed locally:

- Python syntax compilation for packaged source and helper scripts
- Shell syntax checks for entrypoint, host helper, download script, and examples
- YAML parsing of `compose.yaml`
- CLI help checks that do not require the container-only Python dependencies
- Successful help dispatch for every packaged command in the unified entrypoint
- Verification of the two existing SAM weight SHA-256 values
- Resolution check for every pinned direct Python package and an import smoke test
- Verification that the pinned PyTorch base-image tag and amd64 digest exist
- Review that the Docker build context excludes datasets, checkpoints, outputs, and model weights
- Real single-image CPU inference with an existing ViT-B adapter checkpoint; output was a 480x640 grayscale mask with values 0/255
- Real instance-score inference, semantic evaluation, and instance-AP evaluation against the matching ground truth
- Docker Compose image build completed successfully on the target host
- NVIDIA container smoke test passed on one RTX A6000: Python 3.10.8, PyTorch 1.13.1, CUDA 11.6, cuDNN 8302
- Public single-GPU smoke test passed on GPU 0 using the repository's generated synthetic dataset: two training and two validation image/mask pairs (`vit_b`, batch size 1, one epoch)
- Training loss was 0.771236, validation loss was 0.615653, and validation Dice was 0.562301
- The training run produced `args.json`, a TensorBoard event file, and `checkpoint_best.pth`; every artifact was owned by the invoking host user rather than root
- The newly trained checkpoint was loaded in a fresh GPU inference container and produced a 480x640 grayscale mask

The generated smoke-test artifacts and private fixtures are intentionally not included in the public distribution.

Acceptance commands executed successfully:

```bash
docker compose build
docker compose run --rm sam-clip gpu-check
docker compose run --rm sam-clip infer --help
docker compose run --rm sam-clip train --help
./examples/smoke-test-single-gpu.sh
```
