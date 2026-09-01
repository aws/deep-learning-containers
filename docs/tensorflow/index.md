# TensorFlow

TensorFlow DLCs come in two flavors, each with its own guide:

- **[Training](training/index.md)** — train models with the TensorFlow framework on {{ sagemaker }}, with EFA-capable multi-node support on Amazon
  Linux 2023.
- **[Inference](inference/index.md)** — serve TensorFlow SavedModel artifacts with TensorFlow Serving on {{ sagemaker }}, in CPU and GPU variants.

The two are built from separate images and separate ECR repositories (`tensorflow-training` and `tensorflow-inference`), so pick the guide that
matches your workload.
