# EC2 Deployment

The PyTorch DLC is a training image — it does not serve a model out of the box. Launch it on EC2 (or ECS/EKS) with your own training script.

## Single-GPU Training

```bash
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $(pwd):/workspace \
  public.ecr.aws/deep-learning-containers/pytorch:2.12-cu130-amzn2023 \
  python train.py
```

`--shm-size=16g --ipc=host` is required for PyTorch DataLoader workers to share tensors via shared memory.

## Multi-GPU Training (single node)

Use `torchrun` to spawn one process per GPU:

```bash
docker run --rm -it --gpus all --shm-size=16g --ipc=host \
  -v $(pwd):/workspace \
  public.ecr.aws/deep-learning-containers/pytorch:2.12-cu130-amzn2023 \
  torchrun --standalone --nproc_per_node=8 train.py
```

NCCL is pre-configured for multi-GPU collectives — no extra flags required.

## Multi-Node Training (EFA)

For multi-node training on EFA-capable instances (e.g., `p5.48xlarge`, `p4d.24xlarge`), the image ships EFA + the NCCL OFI plugin so collectives flow
over EFA automatically.

Run the container with `--privileged` (or grant the EFA capabilities via `--device`) and pass the EFA devices through, then launch via MPI or
`torchrun`:

```bash
docker run --rm -it --gpus all --privileged --network host \
  --shm-size=16g --ipc=host \
  -v $(pwd):/workspace \
  public.ecr.aws/deep-learning-containers/pytorch:2.12-cu130-amzn2023 \
  torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=demo --rdzv_backend=c10d --rdzv_endpoint=<head_node>:29500 \
    train.py
```

### Verify EFA Connectivity Before Training

The image includes the NCCL `all_reduce_perf` binary at `/usr/local/bin/all_reduce_perf`. Run it across nodes to confirm EFA + NCCL plumbing before
spending GPU-hours on a real job:

```bash
mpirun -np 16 -N 8 -hostfile hosts.txt \
  -x NCCL_DEBUG=INFO -x FI_PROVIDER=efa \
  /usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

## SSH Between Nodes

Multi-node MPI launches require SSH between containers. The image ships a pre-configured OpenSSH server on port 22 that runs as `root` — useful for
test clusters, but you should harden or replace it for production deployments. Expose port 22 with `-p 22:22` (or `--network host`) and add your
public key to `/root/.ssh/authorized_keys`.

## Building on the Image

The image includes `gcc`, `gcc-c++`, `make`, `cuda-nvcc`, and `cuda-cudart-devel`, so you can build CUDA extensions in-place. PyTorch headers and
libraries are visible at `/opt/venv/lib/python3.12/site-packages/torch/`.

## Troubleshooting EFA Throughput

If `all_reduce_perf` runs but throughput is much lower than expected, check that `FI_PROVIDER=efa` is exported (otherwise NCCL falls back to sockets),
and that the NIC is mounted in the container. `lspci | grep -i mellanox` inside the container should list the EFA adapter when EFA is plumbed
correctly.
