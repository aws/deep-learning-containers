# AutoML using AutoGluon DLC

Production-ready Docker images for training and serving [AutoGluon](https://auto.gluon.ai/) models on {{ sagemaker }}. Available in CPU and GPU
variants, built on Amazon Linux 2023 with ongoing security patching.

A single image covers the whole workflow: run a {{ sm_short }} training job with your AutoGluon script, then deploy the resulting model artifact to a
{{ sm_short }} endpoint with the same image. There are no separate training and inference images.

## Images

| Variant | Image |
| --- | --- |
| GPU | `763104351884.dkr.ecr.<region>.amazonaws.com/autogluon:1.6-cu133-amzn2023` |
| CPU | `763104351884.dkr.ecr.<region>.amazonaws.com/autogluon:1.6-cpu-amzn2023` |

Pin the full version (for example `1.6.3-cu133-amzn2023`) to stay on a specific AutoGluon patch release. For other regions and account IDs, see
[Image Access](../get_started/index.md) and [Available Images](../reference/available_images.md).

## What's Included

The images are layered on the [PyTorch {{ sm_short }} DLC](../pytorch/index.md) (PyTorch 2.13, CUDA 13.3 for the GPU variant, Python 3.12) and add:

- **[AutoGluon](https://github.com/autogluon/autogluon) 1.6.3** — `tabular`, `timeseries`, and `multimodal` modules
- **Tabular foundation models** — Mitra, Nori, TabDPT, and TabICL (`autogluon.tabular[mitra,nori,tabdpt,tabicl]`)
- **[Chronos](https://github.com/amazon-science/chronos-forecasting)** — pretrained time series forecasting models
- **Gradient-boosting and classical libraries** — LightGBM, CatBoost, XGBoost, StatsForecast, MLForecast
- **[Ray](https://www.ray.io/)** — parallel model training inside a single job
- **A lightweight inference server** — Gunicorn + Flask on port 8080, implementing the {{ sm_short }} `/ping` and `/invocations` contract

## How It Works

The image entrypoint dispatches on the command {{ sm_short }} passes to the container:

| Command | Behavior |
| --- | --- |
| `train` | Runs your entry script through the {{ sm_short }} training toolkit inherited from the PyTorch DLC (`SM_MODEL_DIR`, `SM_CHANNEL_*`, `SM_NUM_GPUS`, ...) |
| `serve` | Installs `code/requirements.txt` from the model artifact (if present), loads your inference handler, and serves `/ping` and `/invocations` |

Your inference handler lives in the model artifact under `code/` and defines `model_fn` and `transform_fn`. See
[{{ sagemaker }} Deployment](deployment/sagemaker.md) for an end-to-end example and [Configuration](configuration.md) for the handler contract and
environment variables.

## How We Build

- **Built from upstream releases** — images track [AutoGluon releases](https://github.com/autogluon/autogluon/releases) and pin every added package to
  an exact version.
- **Regression-tested** — every release trains and serves tabular and time series models on {{ sm_short }} for both CPU and GPU variants.
- **Security-patched** — continuously maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
