# EKS Deployment

The vLLM container works directly with Kubernetes manifests on Amazon EKS.

## Deployment Example

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
        - name: vllm
          image: public.ecr.aws/deep-learning-containers/vllm:server-cuda
          args:
            - "--model"
            - "openai/gpt-oss-20b"
            - "--host"
            - "0.0.0.0"
            - "--port"
            - "8000"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "1"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 120
```

## Key Requirements

- Request GPU resources via `resources.limits.nvidia.com/gpu`
- Pass `--host 0.0.0.0` so the server binds to all interfaces
- Use `/health` on port 8000 for liveness and readiness probes
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

Add `--tensor-parallel-size 4` to the container args and ensure the node has 4+ GPUs available.

## Model-Specific Tuning

For recommended serving flags, hardware configurations, and quantization options per model, see [recipes.vllm.ai](https://recipes.vllm.ai/).

## Further Reading

- [Deploy LLMs on Amazon EKS using vLLM DLCs](https://aws.amazon.com/blogs/architecture/deploy-llms-on-amazon-eks-using-vllm-deep-learning-containers/)
- [LLM Deployment on Amazon EKS Workshop](https://catalog.us-east-1.prod.workshops.aws/workshops/c22b50fb-64b1-4e18-8d0f-ce990f87eed3/en-US)
