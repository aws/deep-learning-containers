# EC2 Deployment

The Ray LLM DLC is a serving image. On a single-GPU EC2 instance you do not need to bootstrap a Ray cluster — `serve run` starts a local Ray instance,
brings up the vLLM engine from a Ray Serve config, and exposes an OpenAI-compatible endpoint on port 8000.

## The Serve Config

The serving app is configured with a YAML file passed to `serve run`. It sets the HTTP host/port and one `llm_config` — the served model name, where
the weights come from, and the vLLM engine settings. Save this as `config.yaml`:

```yaml
http_options:
  host: 0.0.0.0
  port: 8000
applications:
  - name: llm
    import_path: ray.serve.llm:build_openai_app
    route_prefix: /
    args:
      llm_configs:
        - model_loading_config:
            model_id: ministral
            model_source: mistralai/Ministral-8B-Instruct-2410
          engine_kwargs:
            dtype: bfloat16
            max_model_len: 8192
            gpu_memory_utilization: 0.9
            enable_chunked_prefill: true
            max_num_seqs: 4
          deployment_config:
            autoscaling_config:
              min_replicas: 1
              max_replicas: 1
              target_ongoing_requests: 2
            max_ongoing_requests: 5
```

`model_source` here is the Hugging Face id for [Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410), so the
engine downloads the weights on first start. To serve local or S3-staged weights with no runtime download, set `model_source` to a path such as
`/opt/ml/weights` and bind-mount the model there (shown below). `model_id` (`ministral`) is the name clients pass as `model` and the name `/v1/models`
reports.

## Start the Container

Run this on a GPU EC2 instance with the NVIDIA driver, Docker, and the NVIDIA Container Toolkit installed (a
[Deep Learning AMI](https://aws.amazon.com/ai/machine-learning/amis/) has all three). Ministral-8B fits on a single 24 GB GPU — this example was
verified on a `g6.xlarge`.

Mount the config at `/opt/ml/config/config.yaml` and run `serve run` against it:

```bash
docker run -d --gpus all --shm-size=8g \
  -p 8000:8000 \
  -v $(pwd)/config.yaml:/opt/ml/config/config.yaml:ro \
  public.ecr.aws/deep-learning-containers/ray:serve-llm-cuda \
  serve run /opt/ml/config/config.yaml
```

To serve pre-downloaded weights instead of pulling from Hugging Face, add a read-only weights mount and point `model_source` at it:

```bash
docker run -d --gpus all --shm-size=8g \
  -p 8000:8000 \
  -v /data/ministral:/opt/ml/weights:ro \
  -v $(pwd)/config.yaml:/opt/ml/config/config.yaml:ro \
  public.ecr.aws/deep-learning-containers/ray:serve-llm-cuda \
  serve run /opt/ml/config/config.yaml
```

The model downloads and loads before the endpoint answers, so an early request may be refused — wait and retry. A single `/v1/completions` call
returning `200` is the real readiness signal:

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ministral", "prompt": "ready?", "max_tokens": 1}'
```

## Query the Endpoints

The endpoint is OpenAI-compatible:

**List loaded models** — must include `ministral`:

```bash
curl http://localhost:8000/v1/models
```

**Text completion:**

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "ministral", "prompt": "Hello, how are you?", "max_tokens": 100, "temperature": 0.7}'
```

**Chat completion:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ministral",
    "messages": [{"role": "user", "content": "Explain what a large language model is in one sentence."}],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

`model` must match the `model_id` in the config. Each response carries the OpenAI schema (`id`, `object`, `choices`, `usage`).

For serving a model too large for one GPU, shard it across nodes with KubeRay — see [Amazon EKS Deployment](eks.md).
