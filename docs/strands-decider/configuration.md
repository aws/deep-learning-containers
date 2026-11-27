# Configuration

The container is configured through environment variables, set in the container `Environment` on {{ sagemaker }}. The entrypoint maps them to
`strands-decider serve` options and starts the server on `0.0.0.0:8080`. No variable is required: with none set, the container serves the checkpoint
it finds under `/opt/ml/model` on the GPU.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `DECIDER_CHECKPOINT` | *(unset)* | Checkpoint folder to serve. When unset, the container uses `/opt/ml/model` if it holds a checkpoint, otherwise its one subfolder that does. Set it when `/opt/ml/model` holds more than one checkpoint |
| `DECIDER_DEVICE` | `cuda` | `cuda` or `cpu`. The image is built and tested for GPU instances |
| `DECIDER_WARMUP` | `1` | `0` skips the warm-up at start (`--no-warmup`). See [Warm-Up](#warm-up) |
| `DECIDER_MAX_CONCURRENCY` | `1` | Requests that run on the model at once; the others wait (`--max-concurrency`). `0` removes it (up to 40 at once). See [Concurrency](#concurrency) |
| `DECIDER_DEVICE_MAP` | *(unset)* | `auto` splits a model larger than one GPU across the visible GPUs (`--device-map`). CUDA only |
| `DECIDER_ARGS` | *(empty)* | Any other `strands-decider serve` options (not `--host`, `--port` or `--device`, which the container sets), split like a shell command line, for example `--max-batch 8 --model-name "fraud triage"` |
| `SM_MODEL_DIR` | `/opt/ml/model` | Directory searched for the checkpoint when `DECIDER_CHECKPOINT` is unset |
| `HF_HUB_OFFLINE` | `1` | The container does not contact the Hugging Face Hub. The base model must be in the model data (see [Packaging a Model](models/index.md#packaging-a-model)) |

A checkpoint is a folder that holds `strands_decider_config.json` (or `hobson_config.json`, for checkpoints saved before the rename). When a `base/`
folder sits next to the checkpoint, the checkpoint's config must name it as its base model (`"base_model": "/opt/ml/model/base"`). Otherwise the
container stops at start with a message that names the expected value.

## Common serve Options

These go in `DECIDER_ARGS`. The full list is in the strands-decider
[inference documentation](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md#serve).

| Option | Default | Description |
| --- | --- | --- |
| `--model-name <name>` | the checkpoint folder name | Value returned as `model` in responses and in `/ping` |
| `--max-batch <n>` | `32` | Questions encoded per forward pass. Larger requests run in chunks. Lower it when a very long state with many questions does not fit in GPU memory |
| `--max-batch-tokens <n>` | *(unset)* | Also caps the padded tokens of one forward pass, for long states. The strands-decider documentation suggests `--max-batch 4 --max-batch-tokens 34000` for the Gemma 4 models at a 32,768-token window |
| `--no-prefix-cache` | *(cache on)* | Encode each question with the full state instead of sharing one encoding of the state across the questions |

## Warm-Up

At start, the server loads the model and then answers each question type once and every number of questions up to `--max-batch`, so that no request
waits for a kernel to compile. `/ping` fails until the warm-up is done, so {{ sagemaker }} sends no traffic before then. The model load and warm-up
take about 70 seconds on the Qwen3.5 model. Without the warm-up, the first request on the Qwen3.5 model takes about 37 seconds. Keep it on for
endpoints, and set `ContainerStartupHealthCheckTimeoutInSeconds` to allow for it.

## Concurrency

{{ sagemaker }} real-time endpoints do not cap the number of requests in flight. Without a cap, the server runs up to 40 requests on the model at once
(its worker thread pool), and a burst of large requests can run the GPU out of memory: in the tests, all 64 requests of a burst of 64 large requests
(16 questions each) failed with HTTP 500 on the E4B model on one L40S.

`DECIDER_MAX_CONCURRENCY=1`, the default, runs one request at a time; the others wait without holding a thread or GPU memory. The questions of a
request are already batched on the GPU, so one at a time was also the fastest setting measured (E4B on one L40S): 3.21 requests per second with 18.5
GB of GPU memory, against 2.50 requests per second with 26.2 GB at a cap of 8. With the default, the same burst of 64 requests is answered in full,
and `/ping` keeps answering within the 2 seconds {{ sagemaker }} allows a health check.

Raise the cap only for small requests on a GPU with memory to spare. To serve more traffic, add instances.

## Known Limitations

- **GPU only.** The image targets NVIDIA GPU instances with CUDA kernels compiled for A100, A10G, L4, L40S, and H100. `DECIDER_DEVICE=cpu` is accepted
  but not tested.
- **Offline.** The container does not download models. The model data must include the base model in `base/`.
- **Text only.** Image input (`serve --vision`) is not included.
- **No authentication of its own.** On {{ sagemaker }}, access is controlled by IAM (`sagemaker:InvokeEndpoint`). If you run the container yourself,
  publish its port on the loopback address only (`-p 127.0.0.1:8080:8080`).
- **One model per container.** To serve several models, create one endpoint per model.

## Full Reference

- [strands-decider inference documentation](https://github.com/strands-labs/strands-decider/blob/main/docs/inference.md)
- [strands-decider](https://github.com/strands-labs/strands-decider)
