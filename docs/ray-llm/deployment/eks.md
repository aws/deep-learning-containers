# Amazon EKS Deployment (KubeRay)

To serve a model too large for one GPU, use [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html)'s `RayService`. KubeRay
runs the head and worker pods and applies the Ray Serve config; vLLM's tensor parallelism then shards the model across nodes.

## Prerequisites

- An {{ eks }} cluster with GPU nodes.
- The [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) installed on the cluster — it reconciles the
  `RayService` into head and worker pods.
- The [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) for multi-node serving, so worker pods can
  request `vpc.amazonaws.com/efa`.
- A shared filesystem such as [FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html) for the model cache.

## RayService

The manifest below serves [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) with tensor parallelism 2, placing each shard
on a distinct GPU node. vLLM downloads the weights from the Hub on first start.

```yaml
apiVersion: ray.io/v1
kind: RayService
metadata:
  name: ray-llm
  namespace: ray-llm
spec:
  serviceUnhealthySecondThreshold: 1200
  deploymentUnhealthySecondThreshold: 1200
  serveConfigV2: |
    applications:
      - name: qwen
        import_path: ray.serve.llm:build_openai_app
        route_prefix: /
        args:
          llm_configs:
            - model_loading_config:
                model_id: qwen-14b
                model_source: Qwen/Qwen2.5-14B-Instruct
              engine_kwargs:
                tensor_parallel_size: 2
                max_model_len: 4096
                dtype: bfloat16
                gpu_memory_utilization: 0.85
              deployment_config:
                num_replicas: 1
              placement_group_config:
                bundles:
                  - { CPU: 4, GPU: 1 }
                  - { CPU: 4, GPU: 1 }
                strategy: STRICT_SPREAD
  rayClusterConfig:
    rayVersion: "2.58.0"
    headGroupSpec:
      rayStartParams:
        dashboard-host: "0.0.0.0"
        num-gpus: "0"
      template:
        spec:
          nodeSelector:
            node-type: cpu-ray-llm-head
          tolerations:
            - { key: ray.io/head-node, operator: Equal, value: "true", effect: NoSchedule }
          containers:
            - name: ray-head
              image: public.ecr.aws/deep-learning-containers/ray:serve-llm-cuda
              imagePullPolicy: Always
              env:
                - { name: HF_HOME, value: /fsx/hf-cache }
              ports:
                - { containerPort: 6379, name: gcs }
                - { containerPort: 8265, name: dashboard }
                - { containerPort: 8000, name: serve }
              resources:
                limits:   { cpu: "4", memory: 16Gi }
                requests: { cpu: "2", memory: 8Gi }
              volumeMounts:
                - { name: fsx-storage, mountPath: /fsx }
              lifecycle:
                preStop: { exec: { command: ["/bin/sh", "-c", "ray stop"] } }
          volumes:
            - name: fsx-storage
              persistentVolumeClaim:
                claimName: fsx-lustre-pvc
    workerGroupSpecs:
      - groupName: gpu-workers
        replicas: 2
        minReplicas: 2
        maxReplicas: 2
        rayStartParams:
          num-gpus: "4"
        template:
          spec:
            nodeSelector:
              node-type: g6-ray-llm
            tolerations:
              - { key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule }
              - { key: aws.amazon.com/efa, operator: Equal, value: "true", effect: NoSchedule }
            containers:
              - name: ray-worker
                image: public.ecr.aws/deep-learning-containers/ray:serve-llm-cuda
                imagePullPolicy: Always
                env:
                  - { name: HF_HOME, value: /fsx/hf-cache }
                  - { name: FI_PROVIDER, value: efa }
                  - { name: FI_EFA_USE_DEVICE_RDMA, value: "1" }
                  - { name: FI_EFA_FORK_SAFE, value: "1" }
                  - { name: NCCL_DEBUG, value: INFO }
                  - name: VLLM_HOST_IP
                    valueFrom:
                      fieldRef:
                        fieldPath: status.podIP
                resources:
                  limits:   { nvidia.com/gpu: "4", vpc.amazonaws.com/efa: "1", cpu: "40", memory: 160Gi }
                  requests: { nvidia.com/gpu: "4", vpc.amazonaws.com/efa: "1", cpu: "40", memory: 160Gi }
                volumeMounts:
                  - { name: fsx-storage, mountPath: /fsx }
                  - { name: shm, mountPath: /dev/shm }
                lifecycle:
                  preStop: { exec: { command: ["/bin/sh", "-c", "ray stop"] } }
            volumes:
              - name: fsx-storage
                persistentVolumeClaim:
                  claimName: fsx-lustre-pvc
              - name: shm
                emptyDir: { medium: Memory, sizeLimit: 8Gi }
```

The `serveConfigV2` is the part to preserve when you adapt this manifest: `build_openai_app` exposes the OpenAI-compatible API,
`tensor_parallel_size: 2` splits the model across two GPUs, and the two `STRICT_SPREAD` placement-group bundles force those shards onto separate
nodes.

### Cluster-specific fields to adapt

The following are specific to the example cluster — change them to match yours:

| Field | Example value | Adapt to |
| --- | --- | --- |
| Head `nodeSelector` | `node-type: cpu-ray-llm-head` | your CPU node label |
| Worker `nodeSelector` | `node-type: g6-ray-llm` | your GPU node label |
| `tolerations` | head-node / GPU / EFA taints | the taints your node groups carry |
| Storage | `fsx-lustre-pvc` at `/fsx`, `HF_HOME=/fsx/hf-cache` | your shared filesystem (FSx for Lustre, EFS, or drop it and let each node download to local cache) |

The head pod runs the GCS, dashboard, and Serve proxy, so it needs no GPU (`num-gpus: "0"`). Keep `rayVersion` equal to the Ray in the image.

## Deploy and Query

Apply the manifest and wait for the `RayService` to report `Running` — the Serve app takes several minutes to load the sharded model:

```bash
kubectl apply -f rayservice.yml
kubectl get rayservice ray-llm -n ray-llm -w
```

`RayService` exposes the endpoint through a Kubernetes service (`ray-llm-serve-svc`). Port-forward to it and query the OpenAI-compatible API on port
8000:

```bash
kubectl port-forward -n ray-llm svc/ray-llm-serve-svc 8000:8000
```

```bash
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-14b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

`/v1/models` must list `qwen-14b`, and `/v1/completions` and `/v1/chat/completions` return the OpenAI schema (`id`, `object`, `choices`, `usage`).

## Confirming EFA Is in Use

`NCCL_DEBUG=INFO` is set in the image, so the first collective prints the transport NCCL chose. Once the model is serving, grep a worker's Ray logs:

```bash
WORKER=$(kubectl get pods -n ray-llm -l ray.io/node-type=worker -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n ray-llm "$WORKER" -c ray-worker -- \
  bash -c 'grep -rh "NET/OFI Selected provider" /tmp/ray/session_*/logs' | head -1
```

`NET/OFI Selected provider is efa` confirms EFA is carrying the collectives; a `NET/Socket` line means EFA is not plumbed through.

## Cleanup

Delete the `RayService` to tear down the RayCluster and all head and worker pods:

```bash
kubectl delete rayservice ray-llm -n ray-llm
```

## Single-Node Serving

For a single GPU, deploy the container to EC2 with `docker run` — see [EC2 Deployment](ec2.md). A community sample that runs a single-GPU Ray Serve
deployment on EKS is available at
[aws-samples/sample-aws-deep-learning-containers](https://github.com/aws-samples/sample-aws-deep-learning-containers/tree/main/inference/ray-serve/ray-serve-single-node).
It is a community-maintained example, not an officially supported configuration, so treat it as a starting point.
