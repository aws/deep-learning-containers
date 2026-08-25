# Configuration

The Ray Train DLC has no bespoke configuration layer — there are no `SM_*`-style translation variables and no opinionated entrypoint. You configure it
the way you configure Ray and NCCL: with `ray start` flags, standard Ray environment variables, and NCCL/EFA tuning knobs. This page documents what
the image already sets, so you know what you are overriding.

## Entrypoint Contract

The entrypoint is `/usr/local/bin/entrypoint.sh`. It does exactly one thing before `exec`ing your command: if the host NVIDIA driver is older than the
bundled `cuda-compat` layer, it prepends `/usr/local/cuda/compat` to `LD_LIBRARY_PATH`. No flag or environment variable is needed — the check runs on
every container start.

Because it `exec`s its arguments verbatim, the command you pass is the command that runs:

```bash
docker run ... ray:train-ml-cuda ray start --head --block
docker run ... ray:train-ml-cuda python3 train.py
docker run ... ray:train-ml-cuda                            # CMD default: bash
```

Under KubeRay you should not set a container `command` at all — the operator injects the correct `ray start` invocation for head and worker pods.

## Baked-In Environment

| Variable | Value | Why |
| --- | --- | --- |
| `FI_PROVIDER` | `efa` | Selects the EFA libfabric provider. NCCL falls back to TCP automatically when no EFA adapter is present |
| `FI_LOG_LEVEL` | `warn` | Keeps libfabric quiet unless something is wrong |
| `NCCL_DEBUG` | `INFO` | Prints the provider and interface NCCL selected — the fastest way to confirm EFA is live |
| `CUDA_HOME` | `/usr/local/cuda` | Where extension builds look for the CUDA toolkit |
| `DLC_CONTAINER_TYPE` | `training` | Identifies the image as a training container |
| `PYTHONUNBUFFERED` | `1` | Log lines appear immediately in `kubectl logs` / `docker logs` |

`NCCL_SOCKET_IFNAME` is deliberately **not** set as an environment variable. Instead, `/etc/nccl.conf` carries the exclusion default
`NCCL_SOCKET_IFNAME=^docker0,lo`, which auto-detects the right NIC on any host. Pinning a specific name in the image would break {{ ec2_short }} and
Slurm hosts, whose interfaces are `ens6` / `enp40s0` rather than `eth0`. Override it per platform when you need to — for example, EKS pods should set
`NCCL_SOCKET_IFNAME=eth0`.

`/etc/nccl.conf` also sets `NCCL_DEBUG=INFO`. Environment variables take precedence over the file, so `NCCL_DEBUG=WARN` on the container quiets it.

## Paths

| Path | Contents |
| --- | --- |
| `/opt/venv` | Python 3.13 virtual environment (already first on `PATH`) |
| `/workspace` | `WORKDIR`. Mount your training code here, and use it as `ray job submit --working-dir` |
| `/usr/local/bin/all_reduce_perf` | NCCL test binary for EFA/NCCL validation |
| `/usr/local/cuda` | CUDA 13.0 toolkit, including `nvcc` for building extensions |
| `/opt/amazon/openmpi` | OpenMPI, on `PATH` |
| `/opt/amazon/efa` | EFA installation, on `PATH` |
| `/etc/nccl.conf` | NCCL defaults described above |
| `/tmp/ray` | Ray session logs and object-store spill. Back it with a volume — see [EKS Deployment](deployment/eks.md) |

## Ports

| Port | Purpose | `ray start` flag |
| --- | --- | --- |
| 6379 | GCS on the head; workers connect here | `--port=6379` (head), `--address=<head>:6379` (worker) |
| 8265 | Dashboard and job-submission API | `--dashboard-host=0.0.0.0 --dashboard-port=8265` |
| 10001 | Ray Client server | `--ray-client-server-port=10001` |
| 22 | OpenSSH, for MPI-based launches | n/a |

Port 8000 is **not** exposed — Ray Serve is not part of this image.

## Ray Resource Declaration

Ray schedules against the resources it was told about at `ray start`, not against what the kernel reports. When a container's GPU visibility is
restricted (a Kubernetes GPU limit, or `--gpus '"device=0,1"'`), declare the same number to Ray:

```bash
ray start --address=<head>:6379 --num-gpus=4 --num-cpus=40 --block
```

Under KubeRay this is `rayStartParams.num-gpus`, and it must match the pod's `nvidia.com/gpu` limit. A mismatch means Ray either oversubscribes GPUs
(tasks fail with CUDA OOM or device errors) or leaves them idle.

## Shared Memory

Ray's object store lives in `/dev/shm`. Docker's 64 MB default is far below what Ray Train needs, so always pass `--shm-size` (16 GB is a reasonable
starting point) and `--ipc=host`. On Kubernetes, either mount an `emptyDir` with `medium: Memory` at `/dev/shm` or accept Ray's fallback to disk-based
spill, which is much slower.

## Checkpoint Storage

`RunConfig(storage_path=...)` must resolve to storage every worker can write and the driver can read — a shared filesystem mount (`/fsx`, NFS) or an
`s3://` URI. Ray Train validates this and fails a multi-node run that points at node-local storage. `boto3` and `awscli` are pre-installed, so S3
paths work with the container's instance profile or IRSA role.

## Useful NCCL and EFA Knobs

The defaults are tuned for the common case; reach for these only when diagnosing or tuning:

| Variable | Purpose |
| --- | --- |
| `NCCL_SOCKET_IFNAME` | Pin the interface NCCL uses for bootstrap (`eth0` on EKS pods) |
| `NCCL_DEBUG` | `INFO` by default; `WARN` to quiet, `TRACE` to debug a hang |
| `NCCL_DEBUG_SUBSYS` | Narrow debug output, e.g. `INIT,NET` |
| `FI_EFA_USE_DEVICE_RDMA` | Enable GPUDirect RDMA on instances that support it |
| `FI_PROVIDER` | `efa` by default; set to `tcp` to force a socket fallback for comparison |

## Known Limitations

- **GPU only.** There is no CPU variant and no Trainium/Neuron variant. The image is built against CUDA 13.0 and requires an NVIDIA GPU for training.
- **No Ray Serve.** `ray[serve]` is not installed, and importing `ray.serve` raises `ImportError`. Use the [Ray Serve DLC](../ray/index.md) for
  inference.
- **No {{ sagemaker }} variant.** The image does not include the SageMaker training toolkit and is not wired for SageMaker Training jobs. Targets are
  {{ ec2_short }}, {{ eks_short }}, and HyperPod-EKS.
- **HyperPod-Slurm is not supported yet.** Slurm needs client libraries, `munge`, and PMIx plus a Slurm-to-Ray bootstrap; that target is planned, not
  shipped.
- **No `torchaudio`.** There is no CUDA 13.0 wheel past 2.11.0, and audio I/O is outside the distributed-training scope.
- **No head/worker entrypoint.** The image does not decide whether it is a head or a worker — KubeRay's `RayCluster` spec or your own `ray start`
  command does. This is intentional, so the image stays usable under any orchestrator.
- **SSH runs as root.** The pre-configured OpenSSH server on port 22 exists for MPI launches in test clusters. Harden or disable it for production.
