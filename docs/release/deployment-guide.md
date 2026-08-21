# D3 Release Deployment Guide

This guide prepares a release candidate environment. It does not authorize production deployment or registry promotion.

## Prerequisites

- Python 3.11 on the qualified Windows/CUDA platform.
- NVIDIA driver compatible with CUDA 12.8 and an RTX 5060-class validated GPU.
- Artifact files at the URIs recorded in the candidate manifest.
- No model weights, banks or datasets copied into Git.

## Verify the package

1. Create an isolated Python environment.
2. Install the exact requirements referenced by `dependency-lock.json`. Install backend requirements while the working directory is `backend/`, because its editable `../packages/vision-contract` reference is relative to that directory. Install torch and torchvision from `https://download.pytorch.org/whl/cu128`.
3. Install `packages/vision-contract` from the frozen source tree.
4. Run the release package loader. It must verify the dependency lock, candidate manifest, qualification evidence and every artifact hash before model construction.
5. Run one 1600×256 smoke image and confirm the output contains `image_score`, `anomaly_label`, `heatmap`, `confidence`, `model_version` and `artifact_version`.
6. Run the steel, inference and backend suites.

## Candidate-only start sequence

```text
dependency lock verification
  -> release manifest verification
  -> candidate/artifact hash verification
  -> model load
  -> sealed smoke inference
  -> READY_FOR_MANUAL_RELEASE_REVIEW
```

Any mismatch must stop the sequence. Do not repair a mismatch by editing a hash, threshold or artifact. Restore the exact approved package or execute the rollback procedure.

## Configuration boundaries

- Keep the candidate manifest path explicit; do not point the production registry at this release automatically.
- Do not enable an automatic promotion job.
- Do not expose writable model, bank or whitening paths to the inference service.
- Persist prediction and monitoring logs outside the source tree.
