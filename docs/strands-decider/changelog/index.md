# Changelog

Changelog for the Amazon Linux 2023-based strands-decider images.

* * *

## Strands Decider 1.0.0 (strands-decider 0.3.1.dev1) — 2026-11-10

**Tags:** `serve-sagemaker-cuda` · `serve-sagemaker-cuda-v1` · `serve-sagemaker-cuda-v1.0` · `serve-sagemaker-cuda-v1.0.0`

**strands-decider source:** [`9a7470e`](https://github.com/strands-labs/strands-decider/commit/9a7470eb90e5afe5ae9f0e1dfc52d65bd3ab71b7)

### Highlights

- Initial release of the strands-decider inference container on Amazon Linux 2023, for {{ sagemaker }} real-time endpoints on NVIDIA GPUs (port 8080).
- Serves `strands-decider serve`: `POST /invocations` and `POST /v1/systemone` answer `noul`, `choice`, and `score` questions with calibrated
  probabilities, and `GET /ping` reports readiness.
- Finds the checkpoint under `/opt/ml/model` and loads the base model from `base/` with no network access.
- Warms up before `/ping` succeeds, so the first request does not wait for kernel compiles.
- Runs one request on the model at a time by default (`DECIDER_MAX_CONCURRENCY=1`), so bursts of large requests queue instead of running out of GPU
  memory.
- Splits a model larger than one GPU across the instance's GPUs with `DECIDER_DEVICE_MAP=auto`.
- PyTorch 2.7.1 (CUDA 12.8) on the CUDA 12.9 base image, Python 3.13, transformers 5.19.0, flash-linear-attention, and a compiled causal-conv1d 1.7.0.

### Supported Models at Launch

- `strands-decider-2B-qwen3.5-v1-2610`, `strands-decider-E2B-gemma4-v1-2610`, and `strands-decider-E4B-gemma4-v1-2610`, tested on one L4.
- `strands-decider-12B-gemma4-v1-2610`, tested on one L40S.
- `strands-decider-26B-A4B-gemma4-v1-2610`, tested across L40S GPUs with `DECIDER_DEVICE_MAP=auto`.
