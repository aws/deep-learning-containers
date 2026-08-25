# EC2 Deployment

On {{ ec2_short }} there is no orchestrator to bootstrap the cluster, so you start Ray yourself: one container as the head, the rest as workers
pointing at it. This is the simplest way to try the image, and the right choice when you are not running Kubernetes.

The image's entrypoint runs a CUDA forward-compatibility check and then `exec`s whatever command it is given, so any `ray` invocation works directly
as the container command.

## Single Node, Multi-GPU

For a single node, Ray Train needs no cluster setup — `ray.init()` inside the script starts a local Ray instance that sees every GPU:

```bash
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $(pwd):/workspace \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  python3 train.py
```

`--shm-size=16g --ipc=host` is required — Ray's object store and PyTorch's DataLoader workers both share tensors through `/dev/shm`, and the Docker
default of 64 MB is far too small.

Set `ScalingConfig(num_workers=<gpus_on_this_host>, use_gpu=True)` in the script to use all local GPUs.

## Multi-Node

Start a head container on one instance:

```bash
docker run --rm -d --name ray-head \
  --gpus all --network host --shm-size=16g --ipc=host \
  -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --block
```

Then join each worker instance to it:

```bash
docker run --rm -d --name ray-worker \
  --gpus all --network host --shm-size=16g --ipc=host \
  -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --address=<head_private_ip>:6379 --block
```

`--block` keeps `ray start` in the foreground so the container's lifetime matches the Ray node's. `--network host` lets workers reach the head's GCS
on 6379 without port mapping — the alternative is publishing 6379, 8265, and 10001 explicitly and making sure the security group allows them between
instances.

Confirm the cluster formed, then submit a job:

```bash
docker exec ray-head ray status
docker exec ray-head ray job submit --address http://localhost:8265 --working-dir /shared/code -- python3 train.py
```

Checkpoints and datasets must live on storage every node can reach — an NFS or FSx mount (shown as `/shared` above) or an `s3://` path in
`RunConfig(storage_path=...)`. A host path that exists only on one instance fails as soon as a second node writes a checkpoint.

## Multi-Node with EFA

On EFA-capable instances (for example `p5.48xlarge` or `p4d.24xlarge`) the image already contains EFA, the AWS NCCL OFI plugin, and GDRCopy, so
collectives flow over EFA once the adapter is visible inside the container. Pass the EFA devices through and keep host networking:

```bash
docker run --rm -d --name ray-worker \
  --gpus all --network host --privileged \
  --shm-size=16g --ipc=host \
  -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --address=<head_private_ip>:6379 --block
```

`FI_PROVIDER=efa` and `NCCL_DEBUG=INFO` are already set in the image. `NCCL_SOCKET_IFNAME` is deliberately **not** pinned: `/etc/nccl.conf` ships the
exclusion default `^docker0,lo`, which auto-detects the right NIC on any host. Hardcoding `eth0` would break {{ ec2_short }} instances, whose
interfaces are named `ens6`, `enp40s0`, and similar.

### Verify EFA Connectivity First

`all_reduce_perf` is at `/usr/local/bin/all_reduce_perf`. Run it across nodes before starting a real job:

```bash
docker exec ray-head mpirun -np 16 -N 8 -hostfile /shared/hosts.txt \
  -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa \
  /usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

`NET/OFI Selected provider is efa` in the output confirms the plumbing. `NET/Socket` means NCCL fell back to TCP — check that the container ran with
`--privileged` (or the EFA devices passed via `--device`), and that `lspci | grep -i mellanox` inside the container lists the adapter.

MPI launches need SSH between containers. The image ships an OpenSSH server on port 22 with a root key already generated, which is convenient for test
clusters but should be hardened or replaced for production. Use `--network host` (or `-p 22:22`) and add your public key to
`/root/.ssh/authorized_keys`.

## Building on the Image

`gcc`, `gcc-c++`, `make`, `cuda-nvcc`, and `cuda-cudart-devel` are installed, so CUDA extensions compile in place. Python lives in a venv at
`/opt/venv`, already on `PATH`, with PyTorch headers under `/opt/venv/lib/python3.13/site-packages/torch/`.

## Shutting Down

```bash
docker exec ray-worker ray stop
docker exec ray-head ray stop
docker rm -f ray-worker ray-head
```
