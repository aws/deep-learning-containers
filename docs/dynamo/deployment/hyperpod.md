# Deploy on Amazon SageMaker HyperPod

This guide serves a model with disaggregated prefill and decode on a {{ hyperpod }} cluster orchestrated by {{ eks_short }}, using the Dynamo DLCs and
the HyperPod inference add-on.

## Prerequisites

- A HyperPod cluster with {{ eks_short }} orchestration and the `amazon-sagemaker-hyperpod-inference` add-on installed. The add-on installs the Dynamo
  operator and the `DynamoGraphDeployment` (DGD) custom resource.
- GPU instance groups for the workers. For the EFA images, use EFA-capable instance types; this guide puts prefill and decode on separate nodes.
- `kubectl` access to the cluster.

## Runtime version

The Dynamo operator reads the runtime version from the image tag and turns on version-dependent behavior (health checks, worker failure thresholds)
from it. The DLC tags start with the Dynamo version (`1.6.2-…`) for this reason. To make the version independent of the tag (for a digest reference
`…/dynamo@sha256:…`, or a retagged copy), set it on each component, as the example below does:

```yaml
runtimeVersionOverride: "1.6.2"
```

## Deploy a disaggregated model

This `DynamoGraphDeployment` runs a frontend, one prefill worker and one decode worker with the vLLM EFA image. Replace the instance group names with
your own; HyperPod labels each node with `sagemaker.amazonaws.com/instance-group-name`.

```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: qwen3
  namespace: dynamo-demo
spec:
  components:
    - name: Frontend
      type: frontend
      replicas: 1
      runtimeVersionOverride: "1.6.2"
      podTemplate:
        spec:
          containers:
            - name: main
              image: public.ecr.aws/deep-learning-containers/dynamo:1.6.2-efa-vllm-runtime-v1
    - name: prefill
      type: prefill
      replicas: 1
      runtimeVersionOverride: "1.6.2"
      podTemplate:
        spec:
          nodeSelector:
            sagemaker.amazonaws.com/instance-group-name: <prefill-instance-group>
          containers:
            - name: main
              image: public.ecr.aws/deep-learning-containers/dynamo:1.6.2-efa-vllm-runtime-v1
              command: ["python3", "-m", "dynamo.vllm"]
              args:
                - --model
                - Qwen/Qwen3-0.6B
                - --max-model-len
                - "4096"
                - --disaggregation-mode
                - prefill
                - --kv-transfer-config
                - '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}'
              env:
                - { name: FI_PROVIDER, value: efa }
                - { name: NIXL_LOG_LEVEL, value: INFO }
              resources:
                limits: { nvidia.com/gpu: "1", vpc.amazonaws.com/efa: "1" }
    - name: decode
      type: decode
      replicas: 1
      runtimeVersionOverride: "1.6.2"
      podTemplate:
        spec:
          nodeSelector:
            sagemaker.amazonaws.com/instance-group-name: <decode-instance-group>
          containers:
            - name: main
              image: public.ecr.aws/deep-learning-containers/dynamo:1.6.2-efa-vllm-runtime-v1
              command: ["python3", "-m", "dynamo.vllm"]
              args:
                - --model
                - Qwen/Qwen3-0.6B
                - --max-model-len
                - "4096"
                - --disaggregation-mode
                - decode
                - --kv-transfer-config
                - '{"kv_connector": "NixlConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}'
              env:
                - { name: FI_PROVIDER, value: efa }
                - { name: NIXL_LOG_LEVEL, value: INFO }
              resources:
                limits: { nvidia.com/gpu: "1", vpc.amazonaws.com/efa: "1" }
```

```bash
kubectl create namespace dynamo-demo
kubectl apply -f qwen3-dgd.yaml
kubectl get pods -n dynamo-demo -w
```

The workers download the model from Hugging Face at start. For a gated model, pass a token with `HF_TOKEN` from a Kubernetes secret.

### Other backends and non-EFA images

Change the image and the worker command; the rest of the manifest stays the same.

| Backend | Worker command | KV transfer settings |
| --- | --- | --- |
| vLLM | `python3 -m dynamo.vllm --model <model> --disaggregation-mode <prefill\|decode> --kv-transfer-config '<json>'` | `NixlConnector` with `"backends": ["LIBFABRIC"]` over EFA, `["UCX"]` without |
| SGLang | `python3 -m dynamo.sglang --model-path <model> --served-model-name <model> --page-size 16 --tp 1 --trust-remote-code --disaggregation-mode <prefill\|decode> --disaggregation-transfer-backend nixl --disaggregation-bootstrap-port 12345 --host 0.0.0.0` | env `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` over EFA, `UCX` without |
| TensorRT-LLM | `python3 -m dynamo.trtllm --model-path <model> --served-model-name <model> --extra-engine-args <engine.yaml> --disaggregation-mode <prefill\|decode>` | Over EFA: `cache_transceiver_config.backend: NIXL` in the engine file and env `TRTLLM_NIXL_KVCACHE_BACKEND=LIBFABRIC`. Without EFA: `backend: DEFAULT` |

For the non-EFA images, drop `FI_PROVIDER` and the `vpc.amazonaws.com/efa` resource.

TensorRT-LLM reads its engine settings from a YAML file, which has to exist in the pod: mount it from a ConfigMap, or write it in the container's
command before starting the worker (`sh -c "echo '<yaml>' > /tmp/engine.yaml && exec python3 -m dynamo.trtllm …"`). Prefill and decode need different
files: set `disable_overlap_scheduler: true` for prefill only. NVIDIA's
[engine configs](https://github.com/ai-dynamo/dynamo/tree/v1.6.2/examples/backends/trtllm/engine_configs) are a starting point.

## Send a request

The operator creates a `<name>-frontend` service on port 8000 with the OpenAI API:

```bash
kubectl port-forward -n dynamo-demo svc/qwen3-frontend 8000:8000 &
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
  "max_tokens": 300
}'
```

The frontend can list the model a few seconds before both workers are ready to serve, so the first request can return `503`; retry it.

## Troubleshooting

- **Frontend crash-loops on a CPU node** (`failed to initialize NVML: Driver Not Loaded`): on clusters with CPU instance groups, the HyperPod
  lifecycle scripts may set the NVIDIA container runtime as the default on every node, and CUDA images then fail to start on nodes without a GPU. Set
  `NVIDIA_VISIBLE_DEVICES=void` on the frontend container, or pin it to a GPU instance group.
- **Pods stay `Pending` with `Too many pods`:** HyperPod limits pods per node, and the inference add-on runs about 15 pods of its own. Add nodes to
  the instance groups that run the add-on.
- **Check the KV transfer path:** with `NIXL_LOG_LEVEL=INFO`, the worker logs show which NIXL backend started (`Backend LIBFABRIC was instantiated`
  for vLLM and SGLang over EFA; `using NIXL backend: LIBFABRIC` for TensorRT-LLM).
