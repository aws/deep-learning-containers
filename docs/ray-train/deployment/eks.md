# Amazon EKS Deployment (KubeRay)

KubeRay is the recommended way to run this image. It is the primary target, and the same manifest works on plain {{ eks }} and on SageMaker
HyperPod-EKS — HyperPod adds node health checks and auto-resume *below* the container, so nothing changes inside the image.

## Prerequisites

- An {{ eks }} cluster with GPU nodes, the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin), and — for multi-node training — the
  [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) so pods can request `vpc.amazonaws.com/efa`
- The [KubeRay operator](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) installed in the cluster
- A shared filesystem for datasets and checkpoints — [FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html) via the
  [FSx CSI driver](https://github.com/kubernetes-sigs/aws-fsx-csi-driver) is the usual choice for multi-node training

## RayCluster Manifest

The head pod needs no GPU — it runs the GCS, the dashboard, and the job server. Put the GPUs and EFA devices on the workers:

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: ray-train-cluster
  namespace: ray-train
spec:
  rayVersion: "2.58.0"
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
      num-gpus: "0"
    template:
      spec:
        containers:
          - name: ray-head
            image: public.ecr.aws/deep-learning-containers/ray:train-ml-cuda
            env:
              # Keep the CUDA runtime out of the head pod — it schedules, it does not train.
              - { name: NVIDIA_VISIBLE_DEVICES, value: "void" }
            ports:
              - { containerPort: 6379, name: gcs-server }
              - { containerPort: 8265, name: dashboard }
              - { containerPort: 10001, name: client }
            resources:
              limits:   { cpu: "4", memory: 16Gi }
              requests: { cpu: "4", memory: 16Gi }
            volumeMounts:
              - { name: fsx-storage, mountPath: /fsx }
              - { name: ray-logs, mountPath: /tmp/ray }
            lifecycle:
              preStop: { exec: { command: ["/bin/sh", "-c", "ray stop"] } }
        volumes:
          - { name: ray-logs, emptyDir: {} }
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
          tolerations:
            - { key: nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule }
            - { key: aws.amazon.com/efa, operator: Equal, value: "true", effect: NoSchedule }
          containers:
            - name: ray-worker
              image: public.ecr.aws/deep-learning-containers/ray:train-ml-cuda
              env:
                - { name: NCCL_DEBUG, value: "INFO" }
                - { name: FI_PROVIDER, value: "efa" }
                # On EKS the pod NIC is eth0; pin it so NCCL does not probe other interfaces.
                - { name: NCCL_SOCKET_IFNAME, value: "eth0" }
              resources:
                limits:   { nvidia.com/gpu: "4", vpc.amazonaws.com/efa: "1", cpu: "40", memory: 160Gi }
                requests: { nvidia.com/gpu: "4", vpc.amazonaws.com/efa: "1", cpu: "40", memory: 160Gi }
              volumeMounts:
                - { name: fsx-storage, mountPath: /fsx }
                - { name: ray-logs, mountPath: /tmp/ray }
              lifecycle:
                preStop: { exec: { command: ["/bin/sh", "-c", "ray stop"] } }
          volumes:
            - { name: ray-logs, emptyDir: {} }
            - name: fsx-storage
              persistentVolumeClaim:
                claimName: fsx-lustre-pvc
```

Apply it and wait for the pods:

```bash
kubectl apply -f raycluster.yml
kubectl wait --for=condition=Ready pod \
  -l ray.io/cluster=ray-train-cluster,ray.io/node-type=head \
  -n ray-train --timeout=600s
kubectl get pods -l ray.io/cluster=ray-train-cluster -n ray-train -o wide
```

A few details in the manifest matter:

- **`rayVersion` must match the Ray in the image** (2.58.0). KubeRay uses it to pick default probe endpoints and startup behavior.
- **Do not set a container `command`** — KubeRay injects `ray start` per pod, and overriding it breaks head/worker wiring. The image's entrypoint only
  runs the CUDA forward-compatibility check and then `exec`s whatever it is given.
- **Mount an `emptyDir` at `/tmp/ray`.** Ray writes session logs and spill files there; without a volume they land on the container's writable layer.
- **`preStop: ray stop`** lets Ray drain gracefully instead of being killed mid-job.
- **`num-gpus` in `rayStartParams`** must match the pod's `nvidia.com/gpu` limit, or Ray's scheduler and the kubelet will disagree about capacity.
- **`vpc.amazonaws.com/efa: "1"`** requests the EFA device. Set it only on instance types that have an EFA adapter; the request fails to schedule
  otherwise.

## Submitting a Training Job

Forward the dashboard port and submit from your workstation:

```bash
kubectl port-forward -n ray-train svc/ray-train-cluster-head-svc 8265:8265 &

ray job submit \
  --address http://localhost:8265 \
  --working-dir ./my_training_code \
  -- python3 train.py
```

`--working-dir` is uploaded to the cluster, so keep it scoped to your training code — pointing it at a large directory (or `/`) makes submission slow
or fails outright.

You can also submit from inside the head pod, which avoids the port-forward:

```bash
HEAD_POD=$(kubectl get pod -n ray-train \
  -l ray.io/cluster=ray-train-cluster,ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec "$HEAD_POD" -n ray-train -- \
  ray job submit --address http://localhost:8265 --working-dir /workspace -- python3 train.py
```

## Training Script

A Ray Train script is ordinary PyTorch wrapped in a `TorchTrainer`. `ScalingConfig(num_workers=N, use_gpu=True)` asks Ray for N GPU workers, which Ray
places across the worker pods:

```python
import ray.train
import torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


def train_func(config):
    # Ray serializes train_func by value, so the workers re-run these imports.
    import ray.train.torch
    import torch.nn as nn

    model = ray.train.torch.prepare_model(nn.Linear(32, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    device = next(model.parameters()).device

    for _ in range(config["steps"]):
        x = torch.randn(128, 32, device=device)
        y = torch.randn(128, 1, device=device)
        loss = nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ray.train.report({"loss": loss.item()})


trainer = TorchTrainer(
    train_func,
    train_loop_config={"steps": 100},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    # Checkpoints must go to shared storage so any worker can write them.
    run_config=RunConfig(name="demo", storage_path="/fsx/ray_results"),
)
result = trainer.fit()
print(result.metrics)
```

`prepare_model` wraps the model in DDP and moves it to the worker's GPU. For FSDP with Lightning, use `RayFSDPStrategy` — see the multi-node FSDP
reference in
[test/ray-train/eks/scripts/fsdp_ray.py](https://github.com/aws/deep-learning-containers/blob/main/test/ray-train/eks/scripts/fsdp_ray.py).

`storage_path` must point at shared storage (`/fsx`, or an `s3://` URI) — Ray Train validates that checkpoints written by any worker are readable by
the driver, and a node-local path fails with multiple nodes.

## Offline Datasets

Nodes in a private subnet cannot reach the Hugging Face Hub. Stage the dataset and model onto the shared filesystem once, then run the job offline:

```python
import os

os.environ["HF_HOME"] = "/fsx/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
```

## Verifying EFA Before a Long Run

`all_reduce_perf` ships in the image. Run it inside a worker pod to confirm NCCL selected the EFA provider before committing GPU-hours:

```bash
kubectl exec -n ray-train <worker-pod> -- \
  bash -c 'NCCL_DEBUG=INFO /usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 4'
```

Look for `NET/OFI Selected provider is efa` in the output. If it shows `NET/Socket` instead, EFA is not plumbed through — check that the pod actually
requested `vpc.amazonaws.com/efa`, that the EFA device plugin is running, and that `FI_PROVIDER=efa` is set.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Head pod `Running` but never `Ready` | KubeRay probes the dashboard and GCS endpoints. Confirm ports 6379/8265 are declared and that no container `command` overrides `ray start`. |
| `ray job submit` returns a connection error | The dashboard agent is part of `ray[default]`. Check the port-forward is live and that you targeted the head service, not a worker pod. |
| Worker pods `Pending` forever | GPU or EFA capacity is unavailable. Compare `nvidia.com/gpu` and `vpc.amazonaws.com/efa` requests against the node's allocatable resources, and check tolerations match the node taints. |
| NCCL hangs at the first collective | Usually a network-interface mismatch. Set `NCCL_SOCKET_IFNAME=eth0` on the workers and read the `NCCL_DEBUG=INFO` output for the interface NCCL picked. |
| Checkpointing fails with a path error | `RunConfig(storage_path=...)` points at node-local storage. Move it to `/fsx` or an S3 URI. |
