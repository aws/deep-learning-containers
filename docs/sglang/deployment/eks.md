# EKS Deployment

The SGLang container works directly with Kubernetes manifests on Amazon EKS. It serves the OpenAI-compatible API on port 30000 — the same as
[EC2](ec2.md) — so any `sglang.launch_server` flag may be passed via the container `args`.

## Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sglang-server
  template:
    metadata:
      labels:
        app: sglang-server
    spec:
      containers:
        - name: sglang
          image: public.ecr.aws/deep-learning-containers/sglang:server-cuda
          args:
            - "--model-path"
            - "openai/gpt-oss-20b"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "30000"
          ports:
            - containerPort: 30000
          resources:
            limits:
              nvidia.com/gpu: "1"
          livenessProbe:
            httpGet:
              path: /health
              port: 30000
            initialDelaySeconds: 120
          readinessProbe:
            httpGet:
              path: /health
              port: 30000
            initialDelaySeconds: 120
```

## Key Requirements

- Request GPU resources via `resources.limits.nvidia.com/gpu`
- Pass `--host 0.0.0.0` so the server binds to all interfaces
- Use `/health` on port 30000 for liveness and readiness probes
- Set `initialDelaySeconds` high enough for model loading (120s+ for large models)
- For gated models, provide `HF_TOKEN` via a Kubernetes Secret:

```yaml
env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-secret
        key: token
```

## Multi-GPU

For tensor parallelism across multiple GPUs on a single node:

```yaml
resources:
  limits:
    nvidia.com/gpu: "4"
```

Add `--tp 4` to the container args and ensure the node has 4+ GPUs available.

## Model-Specific Tuning

For recommended serving flags, hardware configurations, and quantization options per model, see the
[SGLang hyperparameter tuning guide](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html) and [Supported Models](../models/index.md).
