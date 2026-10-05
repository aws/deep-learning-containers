# Tabular Prediction and Forecasting using AutoGluon DLC

[AutoGluon](https://auto.gluon.ai/) lets you build accurate ML models for **tabular classification, regression**, and **time series forecasting** with
just a few lines of code. It trains traditional models like XGBoost and CatBoost alongside foundation models like Mitra and Chronos, then combines
them into a single predictor that maximizes performance on your task.

The AutoGluon DLC includes AutoGluon and all the libraries it builds on. Use AutoGluon's API, or train and serve with the underlying packages
directly. The same image runs both {{ sm_short }} training jobs and inference endpoints.

## Images

| Variant | Image |
| --- | --- |
| GPU | `763104351884.dkr.ecr.<region>.amazonaws.com/autogluon:1.6-cu133-amzn2023` |
| CPU | `763104351884.dkr.ecr.<region>.amazonaws.com/autogluon:1.6-cpu-amzn2023` |

To pin a patch release, use the full version (for example `1.6.3-cu133-amzn2023`). For account IDs in other regions, see
[Image Access](../get_started/index.md).

## What's Included

- **[AutoGluon](https://github.com/autogluon/autogluon) 1.6.3**
- **Tabular classification and regression:** [scikit-learn](https://scikit-learn.org/), [LightGBM](https://github.com/microsoft/LightGBM),
  [CatBoost](https://github.com/catboost/catboost), [XGBoost](https://github.com/dmlc/xgboost), [TabM](https://github.com/yandex-research/tabm)
- **Tabular foundation models:** [Mitra](https://huggingface.co/autogluon/mitra-classifier), [TabICL](https://github.com/soda-inria/tabicl),
  [TabDPT](https://github.com/layer6ai-labs/TabDPT), [Nori](https://github.com/synthefy/synthefy-nori)
- **Time series forecasting:** [StatsForecast](https://github.com/Nixtla/statsforecast) (statistical models such as ETS and ARIMA),
  [GluonTS](https://github.com/awslabs/gluonts) (deep learning models such as DeepAR, TFT, and PatchTST),
  [MLForecast](https://github.com/Nixtla/mlforecast)
- **Time series foundation models:** [Chronos](https://github.com/amazon-science/chronos-forecasting),
  [Toto 2.0](https://huggingface.co/collections/Datadog/toto-20)

Foundation model weights are not baked into the image. They are downloaded from Hugging Face the first time a model is used, so the training job or
endpoint needs internet access (or the weights staged locally).

The image is built on the [PyTorch {{ sm_short }} DLC](../pytorch/index.md): PyTorch 2.13, Python 3.12, CUDA 13.3 (GPU variant), Amazon Linux 2023.

## Training and Inference

Run your own training script in a {{ sm_short }} training job, then serve the model from an endpoint with the same image. For inference, define
`model_fn` to load the model and `transform_fn` to make predictions for each request, similar to the now deprecated
[PyTorch Inference DLC](https://sagemaker.readthedocs.io/en/v2/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model).

See [{{ sagemaker }} Deployment](deployment/sagemaker.md) for an end-to-end example and [Configuration](configuration.md) for the handler contract.
