# Amazon EKS Deployment (KubeRay)

[KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started.html) is the usual way to run this image on Kubernetes. Any {{ eks }}
cluster with GPU nodes works. SageMaker HyperPod-EKS is a good fit — it adds node health checks and auto-resume beneath the container, and needs
nothing extra in the image — but it is not required, and the manifest below is unchanged either way. For HyperPod-specific cluster setup, see
[Ray Train on HyperPod-EKS](https://awslabs.github.io/ai-on-sagemaker-hyperpod/docs/eks-orchestration/training-and-fine-tuning/ray-train/ray-train-readme).

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

Four details matter: set `rayVersion` to the Ray in the image; leave the container `command` unset so KubeRay can inject `ray start`; keep
`rayStartParams.num-gpus` equal to the pod's `nvidia.com/gpu` limit so Ray's view of capacity matches the kubelet's; and mount a volume at `/tmp/ray`,
where Ray writes session logs and object-store spill.

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

## Working Reference

[test/ray-train/eks](https://github.com/aws/deep-learning-containers/tree/main/test/ray-train/eks) is this image's multi-node regression test, and it
runs on every release — so unlike a doc snippet, it cannot drift from the image. It is a complete end-to-end example:

| File | What it is |
| --- | --- |
| `raycluster.yml` | The RayCluster manifest the test applies — the same shape as the one above, with the CI cluster's node selectors |
| `scripts/fsdp_ray.py` | A BERT FSDP fine-tune on GLUE/CoLA across 8 GPUs using Ray Train, Ray Data, and PyTorch Lightning |
| `run_eks_test.sh` | The orchestrator: applies the manifest, waits for pods, submits the job, asserts convergence, tears down |

The only thing it assumes that you have to set up yourself is the staged model and dataset, [below](#staging-the-model-and-dataset).

`fsdp_ray.py` covers the pieces most Ray Train jobs need:

- **`ScalingConfig(num_workers=N, use_gpu=True)`** — asks Ray for N GPU workers, which Ray places across the worker pods
- **`RayFSDPStrategy`** with `RayLightningEnvironment` and `RayTrainReportCallback` — FSDP sharding through Lightning
- **`RunConfig(storage_path="/fsx/ray_results")`** — checkpoints on shared storage. This must be a path every worker can write and the driver can
  read, either a shared mount such as `/fsx` or an `s3://` URI; a node-local path breaks checkpointing as soon as a second node writes
- **`HF_HOME=/fsx/hf_cache` and `HF_HUB_OFFLINE=1`** — reads the model and dataset from the shared filesystem instead of downloading them

For plain DDP instead of FSDP, `ray.train.torch.prepare_model()` wraps your model and moves it to the worker's GPU.

### Staging the Model and Dataset

Because that script sets `HF_HUB_OFFLINE=1`, it expects the model and dataset to already be in the HuggingFace cache on the shared filesystem.
Download them once from the head pod before submitting the job:

```bash
kubectl exec "$HEAD_POD" -n ray-train -- env HF_HOME=/fsx/hf_cache python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
AutoTokenizer.from_pretrained('bert-base-cased')
AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=2)
load_dataset('nyu-mll/glue', 'cola')
"
```

Every worker then reads the same cache, and the job needs no internet access from the nodes. Staging up front is worth doing for your own jobs too:
when many workers start at once, downloading in-job makes them all pull the same files simultaneously, and a stalled or rate-limited download can hold
up the first collective until it times out.

If your nodes do have egress and you would rather download at runtime, drop `HF_HUB_OFFLINE=1` and keep `HF_HOME` pointed at the shared mount so the
workers share one cache.

## Confirming EFA Is in Use

`NCCL_DEBUG=INFO` is set in the image, so the first collective of any job prints the transport NCCL chose. Check the job's own output:

```bash
kubectl logs -n ray-train -l ray.io/cluster=ray-train-cluster,ray.io/node-type=worker --tail=-1 | grep -i "NET/OFI\|Libfabric"
```

`NET/OFI Selected provider is efa` confirms EFA is carrying the collectives. A line reporting the `efa` provider returned an empty list, or a fall
back to `NET/Socket`, means EFA is not plumbed through — check that the pod requested `vpc.amazonaws.com/efa`, that the EFA device plugin is running,
and that `NCCL_SOCKET_IFNAME` matches the pod's interface (`eth0` on {{ eks_short }}).
