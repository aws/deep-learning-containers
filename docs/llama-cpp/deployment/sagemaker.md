# Amazon SageMaker AI Deployment

The {{ sagemaker }} image serves on **port 8080** behind an nginx reverse proxy that implements the SageMaker contract: `GET /ping` is mapped to
llama-server's `/health`, and `POST /invocations` is mapped to `/v1/chat/completions`. Every other path is proxied through unchanged, so the full
`/v1/*` OpenAI-compatible API remains reachable. The request and response bodies are the standard OpenAI Chat Completions JSON.

Unlike {{ ec2_short }} — where model and tuning are passed as `llama-server` arguments — the {{ sagemaker }} image is configured entirely through
`SM_LLAMA_CPP_*` environment variables on the container. See [Configuration](../configuration.md).

## Specifying the Model

The {{ sagemaker }} entrypoint resolves the model in this order:

1. **`SM_LLAMA_CPP_MODEL`** — an explicit GGUF path inside the container.
2. **`/opt/ml/model`** — the first `*.gguf` from a `model.tar.gz` staged via `ModelDataUrl` is auto-detected (searched up to two levels deep).
3. **`SM_LLAMA_CPP_HF_REPO` / `SM_LLAMA_CPP_HF_FILE`** — download the GGUF from HuggingFace at container start (requires network egress).

The example below uses the HuggingFace-download path, so nothing needs to be staged in S3. To serve a private or offline model instead, package the
GGUF as a `model.tar.gz`, pass it via `ModelDataUrl`, and drop the `SM_LLAMA_CPP_HF_*` variables.

## Real-Time Endpoint

```python
import boto3
import json

sm = boto3.client("sagemaker")
smrt = boto3.client("sagemaker-runtime")
REGION = boto3.session.Session().region_name

ROLE_ARN = "arn:aws:iam::<account_id>:role/<SageMakerRole>"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/llama-cpp:server-sagemaker-cpu-v1"
NAME = "llama-cpp-realtime"

# 1. Model — configure llama-server through SM_LLAMA_CPP_* env vars
sm.create_model(
    ModelName=NAME,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "Environment": {
            "SM_LLAMA_CPP_HF_REPO": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "SM_LLAMA_CPP_HF_FILE": "qwen2.5-0.5b-instruct-q4_0.gguf",
            "SM_LLAMA_CPP_CTX_SIZE": "4096",
        },
    },
    ExecutionRoleArn=ROLE_ARN,
)

# 2. Endpoint config — allow a generous startup window for model load/download
sm.create_endpoint_config(
    EndpointConfigName=NAME,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": NAME,
        "InitialInstanceCount": 1,
        "InstanceType": "ml.c6i.2xlarge",  # CPU; use ml.c7g.2xlarge for the ARM64 image
        "ContainerStartupHealthCheckTimeoutInSeconds": 600,
    }],
)

# 3. Endpoint
sm.create_endpoint(EndpointName=NAME, EndpointConfigName=NAME)
sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)

# 4. Invoke — POST /invocations is routed to /v1/chat/completions
payload = json.dumps({
    "messages": [{"role": "user", "content": "In one sentence, what is llama.cpp?"}],
    "max_tokens": 64,
})
resp = smrt.invoke_endpoint(EndpointName=NAME, ContentType="application/json", Body=payload)
body = json.loads(resp["Body"].read())
print(body["choices"][0]["message"]["content"])

# 5. Cleanup
sm.delete_endpoint(EndpointName=NAME)
sm.delete_endpoint_config(EndpointConfigName=NAME)
sm.delete_model(ModelName=NAME)
```

To deploy the **Graviton** image, swap `IMAGE_URI` to
`763104351884.dkr.ecr.{REGION}.amazonaws.com/llama-cpp-arm64:server-sagemaker-cpu-v1` and use an ARM64 instance such as `ml.c7g.2xlarge`. For the
**GPU** image, use `llama-cpp:server-sagemaker-cuda-v1`, a GPU instance (e.g. `ml.g6.2xlarge`), add `SM_LLAMA_CPP_N_GPU_LAYERS: "999"` to offload all
layers, and pin a GPU driver AMI — see [Notes](#notes).

## Streaming

`llama-server` supports server-sent-events streaming, and the nginx proxy is configured to pass it through (`proxy_buffering off`). Set
`"stream": true` in the payload and call `invoke_endpoint_with_response_stream` to receive tokens incrementally.

## Notes

- **Configuration is via `SM_LLAMA_CPP_*` env vars.** Any `SM_LLAMA_CPP_FOO_BAR=value` becomes `llama-server --foo-bar value` (a value of `true`
  becomes a bare flag; `false` is omitted). `SM_LLAMA_CPP_MODEL`, `SM_LLAMA_CPP_MODEL_DIR`, and `SM_LLAMA_CPP_PORT` are handled specially — see
  [Configuration](../configuration.md).
- **GPU variants need a recent driver AMI.** The GPU image ships the CUDA 13.0.2 runtime, which is newer than the default {{ sagemaker }} host AMI
  driver. Pin a recent GPU inference AMI via `InferenceAmiVersion` (for example `al2-ami-sagemaker-inference-gpu-3-1` or newer) on the production
  variant, and set `SM_LLAMA_CPP_N_GPU_LAYERS` to offload layers.
- **Generous startup timeout.** The server binds only after the model finishes loading (or downloading), so set
  `ContainerStartupHealthCheckTimeoutInSeconds` to at least 600 seconds; larger models or slow downloads may need more.

For all configuration options, see [Configuration](../configuration.md).
