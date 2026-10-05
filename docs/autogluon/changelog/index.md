# Changelog

Changelog for the Amazon Linux 2023-based AutoGluon images.

* * *

## AutoGluon 1.6.3 — 2026-10-02

**Tags:** `1.6.3-cu133-amzn2023` · `1.6-cu133-amzn2023` · `1.6.3-cpu-amzn2023` · `1.6-cpu-amzn2023`

**AutoGluon source:** [v1.6.3](https://github.com/autogluon/autogluon/releases/tag/v1.6.3)

### Highlights

- One image for {{ sm_short }} training and inference, replacing the separate `autogluon-training` and `autogluon-inference` images (AutoGluon 1.5 and
  earlier).
- Built on the PyTorch 2.13 {{ sm_short }} DLC (Amazon Linux 2023, Python 3.12, CUDA 13.3).
- Adds the Mitra, TabICL, TabDPT, and Nori tabular foundation models and the Chronos and Toto 2.0 time series foundation models.

### Migrating from 1.5

- Use the `autogluon` image for both training jobs and inference endpoints.
- Inference handlers must define `model_fn` and `transform_fn`. `input_fn`, `predict_fn`, and `output_fn` are not called; move that logic into
  `transform_fn`.
