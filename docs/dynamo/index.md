# Disaggregated LLM Serving using NVIDIA Dynamo DLCs

Docker images for running [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) with one inference backend (vLLM, SGLang, or TensorRT-LLM) on
Kubernetes, built for {{ hyperpod }} clusters orchestrated by {{ eks_short }}. Built on Amazon Linux 2023 with ongoing security patching.

Dynamo serves a model as a graph: a frontend that speaks the OpenAI API, and workers that run the backend. In disaggregated serving, prefill and
decode run in separate workers, often on separate nodes, and the KV cache moves between them through NIXL, over EFA on EFA-capable instances. On
HyperPod, the inference add-on installs the Dynamo operator, and you describe the graph as a `DynamoGraphDeployment` (see
[Deployment](deployment/hyperpod.md)).

## Images

Six images, one per backend, each with and without EFA:

| Backend | EFA | Image |
| --- | --- | --- |
| vLLM 0.31.0 | No | `public.ecr.aws/deep-learning-containers/dynamo:1.6.2-vllm-runtime-v1` |
| vLLM 0.31.0 | Yes | `public.ecr.aws/deep-learning-containers/dynamo:1.6.2-efa-vllm-runtime-v1` |
| SGLang 0.5.21 | No | `public.ecr.aws/deep-learning-containers/dynamo:1.6.2-sglang-runtime-v1` |
| SGLang 0.5.21 | Yes | `public.ecr.aws/deep-learning-containers/dynamo:1.6.2-efa-sglang-runtime-v1` |
| TensorRT-LLM 1.3.0rc29 | No | `public.ecr.aws/deep-learning-containers/dynamo:1.6.2-trtllm-runtime-v1` |
| TensorRT-LLM 1.3.0rc29 | Yes | `public.ecr.aws/deep-learning-containers/dynamo:1.6.2-efa-trtllm-runtime-v1` |

All images are x86_64, for NVIDIA GPUs. They are also on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/dynamo). For
private ECR URIs, see [Image Access](../get_started/index.md).

Use an `-efa-` image on EFA-capable instances, where prefill and decode on different nodes transfer the KV cache over EFA. Without EFA, vLLM and
SGLang transfer it with NIXL over UCX, and TensorRT-LLM with its own cache transceiver.

## Tags

Every tag starts with the Dynamo version, because the Dynamo operator reads the runtime version from the image tag (see
[Runtime version](deployment/hyperpod.md#runtime-version)). After it come the backend, the EFA flavor, and the image version:

| Tag | Updates |
| --- | --- |
| `1.6.2-efa-vllm-runtime-v1` | New features and fixes for Dynamo 1.6.2, no breaking changes |
| `1.6.2-efa-vllm-runtime-v1.0` | Security patches and bug fixes only |
| `1.6.2-efa-vllm-runtime-v1.0.1` | Immutable snapshot (`v1.0.0` was the first release) |
| `1.6.2-efa-vllm-runtime` | The latest image for Dynamo 1.6.2 |

The same scheme applies to every backend and flavor. Each tag also has a `-soci` variant with a SOCI index for lazy loading. The images are rebuilt
and released weekly with security patches, as a new patch version.

## What's Included

- **Dynamo 1.6.2**, built from source at the release commit: `ai-dynamo`, `ai-dynamo-runtime` (frontend, router, and the backend components), and
  `gpu-memory-service`
- **One backend** at the exact version Dynamo 1.6.2 supports: vLLM 0.31.0, SGLang 0.5.21, or TensorRT-LLM 1.3.0rc29, in a single locked Python 3.12
  environment
- **NIXL** for KV cache transfer between prefill and decode: its UCX plugin in every image, and its LIBFABRIC (EFA) plugin in the EFA images
- **EFA images:** the EFA installer (libfabric, aws-ofi-nccl) and nccl-tests
- **CUDA 13.0** runtime on Amazon Linux 2023

Not included: NATS and etcd (the operator uses Kubernetes for discovery), KVBM, and Dynamo's Planner and its `aisimulate` dependency.

## Entry Points

The images have no fixed entrypoint; each graph component runs its own command:

| Component | Command |
| --- | --- |
| Frontend (OpenAI API on port 8000) | `python3 -m dynamo.frontend` |
| vLLM worker | `python3 -m dynamo.vllm --model <model> ...` |
| SGLang worker | `python3 -m dynamo.sglang --model-path <model> ...` |
| TensorRT-LLM worker | `python3 -m dynamo.trtllm --model-path <model> ...` |

## Validation

Every image is tested before release on a {{ hyperpod }} cluster with the inference add-on, with the deployment in
[Deploy on Amazon SageMaker HyperPod](deployment/hyperpod.md) for its backend: prefill and decode on two GPU nodes, KV transfer over EFA (EFA images)
or the non-EFA path, chat requests through the frontend, and a check of which NIXL backend the workers started.
