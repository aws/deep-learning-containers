# Serving Decision Models using Strands Decider DLC

Docker image for serving [strands-decider](https://github.com/strands-labs/strands-decider) models on {{ sagemaker }} real-time endpoints. Built on
Amazon Linux 2023 with ongoing security patching.

strands-decider models are small decision models. They do not generate text. Given a `state` (the text to judge) and a set of typed questions, they
return a calibrated probability distribution for each question: a yes/no probability, a choice among named options, or a score on an ordered scale.
The published models are adapters on Qwen3.5 2B and on Gemma 4 E2B, E4B, 12B and 26B-A4B (see [Supported Models](models/index.md)).

## Images

| Platform | Architecture | Device | Image |
| --- | --- | --- | --- |
| {{ sagemaker }} | x86_64 | GPU (CUDA) | `public.ecr.aws/deep-learning-containers/strands-decider:serve-sagemaker-cuda-v1` |

The image is also available on the [ECR Public Gallery](https://gallery.ecr.aws/deep-learning-containers/strands-decider). For private ECR URIs, see
[Image Access](../get_started/index.md).

The tags follow the image version (1.0.0 for the first release):

| Tag | Updates |
| --- | --- |
| `serve-sagemaker-cuda-v1` | New features and fixes, no breaking changes |
| `serve-sagemaker-cuda-v1.0` | Security patches and bug fixes only |
| `serve-sagemaker-cuda-v1.0.0` | Immutable snapshot |
| `serve-sagemaker-cuda` | The latest release |

## What's Included

- **strands-decider** 0.3.1.dev1, built from commit
  [`9a7470e`](https://github.com/strands-labs/strands-decider/commit/9a7470eb90e5afe5ae9f0e1dfc52d65bd3ab71b7) (v0.3.0 plus
  `serve --max-concurrency`). The default entrypoint runs `strands-decider serve` on port 8080.
- **PyTorch 2.7.1** (CUDA 12.8 build) on the CUDA 12.9 Amazon Linux 2023 base image, with Python 3.13 and transformers 5.19.0
- **flash-linear-attention** and a compiled **causal-conv1d** 1.7.0, the CUDA kernels for the Gated DeltaNet layers of the Qwen3.5 torso, built for
  A100, A10G, L4, L40S, and H100 GPUs
- **CUDA forward compatibility**, enabled at start on hosts whose driver is older than the image's CUDA

The model is not in the image. {{ sagemaker }} copies it to `/opt/ml/model` before the container starts, and the container loads it with no network
access (`HF_HUB_OFFLINE=1`).

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /invocations` | Answer the questions in a request ({{ sm_short }} path) |
| `POST /v1/systemone` | The same handler under the strands-decider path |
| `GET /ping` | Readiness check ({{ sm_short }} path). Returns the served model, device, and window once the model is loaded and warmed up |
| `GET /health` | The same as `/ping` |

A request names a `state` and one or more questions, each with a `type` of `noul`, `choice`, or `score`:

```json
{
  "state": "Help! My payouts have been failing for 3 days and nobody answers my emails!",
  "questions": {
    "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {
      "type": "choice",
      "instructions": "Route it.",
      "criteria": {"billing": "money, payouts, invoices", "technical": "bugs", "sales": "pricing"}
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
```

The response holds one answer per question name, plus the served model name, token usage, and latency:

| Type | Answer fields |
| --- | --- |
| `noul` | `noul`: the probability that the answer is yes, from 0 to 1 |
| `choice` | `choice` (the most likely option), `probabilities` (one per option, summing to 1), `confidence` |
| `score` | `score` (the expected level, 0 to levels − 1), `legend`, `probabilities` (one per level), `confidence` |

A malformed question (for example a `choice` without `criteria`) is rejected with HTTP 422. The full request format, including structured states and
option descriptions, is in the strands-decider
[inference documentation](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md#serve). See
[{{ sagemaker }} Deployment](deployment/sagemaker.md) for an end-to-end example and [Configuration](configuration.md) for the launch options.

## How We Build

- **Pinned upstream** — the image installs strands-decider from a pinned commit, with a locked Python environment.
- **Regression-tested** — every release is unit-tested in the image, tested on GPUs with every published model, and on a {{ sagemaker }}
  real-time endpoint. See [Supported Models](models/index.md) for the GPUs.
- **Security-patched** — maintained with security patches from {{ aws }} on an Amazon Linux 2023 base.
