# Changelog

Changelog for the Amazon Linux 2023-based AutoGluon images.

* * *

## AutoGluon 1.6.3 — 2026-10-02

**Tags:** `1.6.3-cu133-amzn2023` · `1.6-cu133-amzn2023` · `1.6.3-cpu-amzn2023` · `1.6-cpu-amzn2023`

**AutoGluon source:** [v1.6.3](https://github.com/autogluon/autogluon/releases/tag/v1.6.3)

### Highlights

- Unified image for {{ sm_short }} training and inference. Replaces the separate `autogluon-training` and `autogluon-inference` images used up to
  AutoGluon 1.5.
- New `autogluon` ECR repository with tags of the form `<version>-<cpu|cuda>-amzn2023`.
- Built on the PyTorch 2.13 {{ sm_short }} DLC: Amazon Linux 2023, CUDA 13.3 (GPU variant), Python 3.12.
- Includes the Mitra, Nori, TabDPT, and TabICL tabular foundation models and Chronos for time series forecasting.
- New lightweight inference server that loads a `model_fn` / `transform_fn` handler from the model artifact.

### Migrating from 1.5 and Earlier

- Use the same `autogluon` image URI for both the training job and the model. The image picks the right mode from the `train` / `serve` command.
- Inference handlers keep the `model_fn` / `transform_fn` contract. `input_fn`, `predict_fn`, and `output_fn` are not supported; fold that logic into
  `transform_fn`.
- Multi-model endpoints are not supported.
