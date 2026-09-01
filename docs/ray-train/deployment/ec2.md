# EC2 Deployment

The Ray Train DLC is a training image. Unlike a one-shot `torchrun` launch, Ray needs a cluster: you start one container as the head and the rest as
workers pointing at it, then submit jobs to the head.

## Single-GPU Training

On one instance Ray needs no cluster setup — `ray.init()` inside the script starts a local Ray instance:

```bash
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $(pwd):/workspace \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  python3 train.py
```

`--shm-size=16g --ipc=host` is required — Ray's object store and PyTorch's DataLoader workers both share memory through `/dev/shm`.

## Multi-GPU Training (single node)

The container command is unchanged. Ray spreads workers over the GPUs it can see, so scale by raising `num_workers` in the script rather than by
wrapping the launch in `torchrun`:

```python
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

trainer = TorchTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=8, use_gpu=True),
)
```

Ray Train handles process placement and the NCCL process group; `ray.train.torch.prepare_model()` wraps your model in DDP.

## Multi-Node Training (EFA)

For multi-node training on EFA-capable instances (e.g., `p5.48xlarge`, `p4d.24xlarge`), the image ships EFA + the NCCL OFI plugin so collectives flow
over EFA automatically.

Run the containers with `--privileged` (or grant the EFA capabilities via `--device`) and use `--network host` so the workers can reach the head's
GCS. Start the head on one instance:

```bash
docker run -d --name ray-head --gpus all --privileged --network host \
  --shm-size=16g --ipc=host -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --block
```

Then join each worker to it:

```bash
docker run -d --name ray-worker --gpus all --privileged --network host \
  --shm-size=16g --ipc=host -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --address=<head_private_ip>:6379 --block
```

`--block` keeps `ray start` in the foreground so the container's lifetime matches the Ray node's. Confirm the cluster formed, then submit a job:

```bash
docker exec ray-head ray status
docker exec ray-head ray job submit --address http://localhost:8265 --working-dir /shared/code -- python3 train.py
```

Checkpoints and datasets must live on storage every node can reach — an NFS or FSx mount (`/shared` above) or an `s3://` path in
`RunConfig(storage_path=...)`.

### Verify EFA Connectivity Before Training

The image includes the NCCL `all_reduce_perf` binary at `/usr/local/bin/all_reduce_perf`. Run it across nodes to confirm EFA + NCCL plumbing before
spending GPU-hours on a real job:

```bash
docker exec ray-head mpirun -np 16 -N 8 -hostfile /shared/hosts.txt \
  -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa \
  /usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

`NET/OFI Selected provider is efa` in the output confirms EFA is carrying the collectives.

## SSH Between Nodes

Multi-node MPI launches require SSH between containers. The image ships a pre-configured OpenSSH server. Start it with `/usr/sbin/sshd`, run it on a
spare port such as 2022 (`--network host` leaves port 22 to the host), and add your public key to `/root/.ssh/authorized_keys`. The default
configuration permits `root` login, so restrict access to your cluster's security group and supply your own keys and hardening.

## Building on the Image

The image includes `gcc`, `gcc-c++`, `make`, `cuda-nvcc`, and `cuda-cudart-devel`, so you can build CUDA extensions in-place. PyTorch headers and
libraries are visible at `/opt/venv/lib/python3.13/site-packages/torch/`.

## Troubleshooting EFA Throughput

If `all_reduce_perf` runs but throughput is much lower than expected, check that `FI_PROVIDER=efa` is exported (otherwise NCCL falls back to sockets),
and that the NIC is mounted in the container. `lspci | grep -i mellanox` inside the container should list the EFA adapter when EFA is plumbed
correctly.
