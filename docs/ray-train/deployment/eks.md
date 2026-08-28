# Amazon EKS Deployment (KubeRay)

[KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) is the usual way to run this image on Kubernetes. Any {{ eks }}
cluster with GPU nodes works. SageMaker HyperPod-EKS is a good fit — it adds node health checks and auto-resume beneath the container, and needs
nothing extra in the image — but it is not required, and the manifest below is unchanged either way.

For multi-node training, install the [EFA device plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) so pods can
request `vpc.amazonaws.com/efa`, and provide a shared filesystem such as
[FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html) for datasets and checkpoints.

## RayCluster

The head pod runs the GCS, dashboard, and job server, so it needs no GPU. Put the GPUs and EFA devices on the workers:

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
              # Skip the NVML hook on this GPU-less pod.
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
      rayStartParams:
        num-gpus: "4"
      template:
        spec:
          containers:
            - name: ray-worker
              image: public.ecr.aws/deep-learning-containers/ray:train-ml-cuda
              env:
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

Four details matter: `rayVersion` must match the Ray in the image; leave the container `command` unset so KubeRay can inject `ray start`; keep
`rayStartParams.num-gpus` equal to the pod's `nvidia.com/gpu` limit, or Ray's scheduler and the kubelet will disagree about capacity; and mount a
volume at `/tmp/ray`, where Ray writes session logs and object-store spill.

`NVIDIA_VISIBLE_DEVICES=void` on the head is worth keeping. When a cluster's default containerd runtime is `nvidia`, the NVML hook runs on every pod
and hard-fails on a GPU-less node with `failed to initialize NVML: Driver Not Loaded`. Setting it to `void` tells the runtime to skip the hook for
that pod.

Apply it and wait for the pods:

```bash
kubectl apply -f raycluster.yml
kubectl wait --for=condition=Ready pod \
  -l ray.io/cluster=ray-train-cluster,ray.io/node-type=head \
  -n ray-train --timeout=600s
```

## Submitting a Training Job

Submit from inside the head pod:

```bash
HEAD_POD=$(kubectl get pod -n ray-train \
  -l ray.io/cluster=ray-train-cluster,ray.io/node-type=head \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec "$HEAD_POD" -n ray-train -- \
  ray job submit --address http://localhost:8265 --working-dir /workspace -- python3 train.py
```

`--working-dir` is uploaded to the cluster, so keep it scoped to your training code. To submit from your workstation instead, run
`kubectl port-forward -n ray-train svc/ray-train-cluster-head-svc 8265:8265` and point `--address` at `http://localhost:8265`.

## Training Script

`ScalingConfig(num_workers=N, use_gpu=True)` asks Ray for N GPU workers, which Ray places across the worker pods:

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

    # Every worker must call report the same number of times — report once per epoch, not per step.
    ray.train.report({"loss": loss.item()})


trainer = TorchTrainer(
    train_func,
    train_loop_config={"steps": 100},
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
    run_config=RunConfig(name="demo", storage_path="/fsx/ray_results"),
)
print(trainer.fit().metrics)
```

`prepare_model` wraps the model in DDP and moves it to the worker's GPU. `storage_path` must point at storage every worker can write and the driver
can read — a shared mount such as `/fsx`, or an `s3://` URI — since Ray Train fails a multi-node run that checkpoints to node-local storage. For FSDP
with Lightning, use `RayFSDPStrategy`; our multi-node regression test is a working example:
[test/ray-train/eks](https://github.com/aws/deep-learning-containers/tree/main/test/ray-train/eks).

If your nodes have no route to the internet, pre-stage the dataset and model onto the shared filesystem and set `HF_HOME=/fsx/hf_cache` plus
`HF_HUB_OFFLINE=1` in the training script.

## Confirming EFA Is in Use

`NCCL_DEBUG=INFO` is set in the image, so the first collective of any job prints the transport NCCL chose. Check the job's own output:

```bash
kubectl logs -n ray-train -l ray.io/cluster=ray-train-cluster,ray.io/node-type=worker --tail=-1 | grep -i "NET/OFI\|Libfabric"
```

`NET/OFI Selected provider is efa` confirms EFA is carrying the collectives. A line reporting the `efa` provider returned an empty list, or a fall
back to `NET/Socket`, means EFA is not plumbed through — check that the pod requested `vpc.amazonaws.com/efa`, that the EFA device plugin is running,
and that `NCCL_SOCKET_IFNAME` matches the pod's interface (`eth0` on {{ eks_short }}).
