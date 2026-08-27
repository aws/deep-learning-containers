# EC2 Deployment

There is no orchestrator on {{ ec2_short }} to bootstrap the cluster, so you start Ray yourself: one container as the head, the rest as workers
pointing at it. The image's entrypoint `exec`s whatever command it is given, so `ray` and `python3` both work directly as the container command.

## Single-Node Training

On one instance Ray needs no cluster setup — `ray.init()` inside the script starts a local Ray instance that sees every GPU:

```bash
docker run --rm -it --gpus all --shm-size=16g \
  -v $(pwd):/workspace \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  python3 train.py
```

Ray's object store lives in `/dev/shm`, and Docker's 64 MB default is far too small — pass `--shm-size`, or mount the host's with
`-v /dev/shm:/dev/shm`. Set `ScalingConfig(num_workers=<gpus_on_this_host>, use_gpu=True)` in the script to use all local GPUs.

## Multi-Node Training (EFA)

On EFA-capable instances (e.g., `p5.48xlarge`, `p4d.24xlarge`) the image ships EFA, the AWS NCCL OFI plugin, and GDRCopy, so collectives flow over EFA
once the adapter is visible in the container. Pass the EFA devices through, raise the memlock limit for pinned RDMA buffers, and use host networking.

Start the head on one instance:

```bash
docker run -d --name ray-head --runtime=nvidia --gpus all \
  --network host --ulimit memlock=-1:-1 \
  $(for d in /dev/infiniband/uverbs*; do echo -n "--device $d "; done) \
  -v /dev/shm:/dev/shm -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --head --port=6379 --dashboard-host=0.0.0.0 --block
```

Then join each worker to it, with the same flags:

```bash
docker run -d --name ray-worker --runtime=nvidia --gpus all \
  --network host --ulimit memlock=-1:-1 \
  $(for d in /dev/infiniband/uverbs*; do echo -n "--device $d "; done) \
  -v /dev/shm:/dev/shm -v /shared:/shared \
  public.ecr.aws/deep-learning-containers/ray:train-ml-cuda \
  ray start --address=<head_private_ip>:6379 --block
```

`--block` keeps `ray start` in the foreground so the container's lifetime matches the Ray node's. `--network host` lets workers reach the head's GCS
on 6379 without port mapping; otherwise publish 6379, 8265, and 10001 and allow them between instances in the security group.

Confirm the cluster formed, then submit:

```bash
docker exec ray-head ray status
docker exec ray-head ray job submit --address http://localhost:8265 --working-dir /shared/code -- python3 train.py
```

Checkpoints and datasets must live on storage every node can reach — an NFS or FSx mount (`/shared` above) or an `s3://` path in
`RunConfig(storage_path=...)`.

### Verify EFA Connectivity Before Training

The image includes the NCCL `all_reduce_perf` binary. Run it across nodes to confirm EFA + NCCL plumbing before spending GPU-hours on a real job:

```bash
docker exec ray-head mpirun -x FI_PROVIDER=efa -x FI_EFA_FORK_SAFE=1 \
  -n 16 -N 8 --hostfile /shared/hosts.txt \
  -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=^lo \
  --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
  /usr/local/bin/all_reduce_perf -b 8 -e 1G -f 2 -g 1
```

Add `-x FI_EFA_USE_DEVICE_RDMA=1` on p4d and p5 instances. `NET/OFI Selected provider is efa` and `Using network Libfabric` in the output confirm the
plumbing; `NET/Socket` means NCCL fell back to TCP.

## SSH Between Nodes

Multi-node MPI launches require SSH between containers. The image ships a pre-configured OpenSSH server on port 22 that runs as `root` — useful for
test clusters, but you should harden or replace it for production deployments. Use `--network host` (or `-p 22:22`) and add your public key to
`/root/.ssh/authorized_keys`.

## Building on the Image

The image includes `gcc`, `gcc-c++`, `make`, `cuda-nvcc`, and `cuda-cudart-devel`, so you can build CUDA extensions in-place. PyTorch headers and
libraries are visible at `/opt/venv/lib/python3.13/site-packages/torch/`.
